/****************************************************************************
 * Copyright (c) 2024 by Oak Ridge National Laboratory                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of Finch. Finch is distributed under a                 *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Solver.hpp
  \brief Main class for heat transport solve
*/

#ifndef Solver_H
#define Solver_H

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

namespace Finch
{

struct HostTag
{
};
struct DeviceTag
{
};

template <typename ViewType, typename EntityType, typename LocalMeshType>
class Solver
{
  protected:
    // temperature views are default constructed and updated every step.
    ViewType T_;
    ViewType T0_;

    LocalMeshType local_mesh_;

    // solution parameters
    double dt_;
    double solidus_;
    double liquidus_;
    double inv_freezing_range_;
    double rho_cp_;
    double rho_Lf_by_dT_;
    double inv_dx2_;
    double k_solid_0_;
    double k_solid_T_;
    double k_liquid_0_;
    double k_liquid_T_;
    double temp_max_;

    // heat source parameters
    double power_;
    double position_[3];
    double r_[3];
    double A_inv_[3];
    double I0_;
    double w_max_;

  public:
    Solver( Inputs db, LocalMeshType local_mesh )
        : local_mesh_( local_mesh )
        , power_( 0.0 )
    {
        // solution parameter constants
        double dx = db.space.cell_size;
        double rho = db.properties.density;
        double cp = db.properties.specific_heat;
        double Lf = db.properties.latent_heat;

        dt_ = db.time.time_step;

        solidus_ = db.properties.solidus;

        liquidus_ = db.properties.liquidus;

        inv_freezing_range_ = 1.0 / ( liquidus_ - solidus_ );

        temp_max_ = db.properties.vaporization_temperature;

        rho_cp_ = rho * cp;

        rho_Lf_by_dT_ = rho * Lf / ( liquidus_ - solidus_ );

        inv_dx2_ = 1.0 / ( dx * dx );

        k_solid_0_ = db.properties.thermal_conductivity_solid_0;
        k_solid_T_ = db.properties.thermal_conductivity_solid_T;
        k_liquid_0_ = db.properties.thermal_conductivity_liquid_0;
        k_liquid_T_ = db.properties.thermal_conductivity_liquid_T;

        // initialize beam position
        for ( std::size_t d = 0; d < 3; ++d )
        {
            position_[d] = 0.0;
        }

        // heat source parameter constants
        for ( std::size_t d = 0; d < 3; ++d )
        {
            r_[d] = db.source.two_sigma[d] / Kokkos::sqrt( 2.0 );
            A_inv_[d] = 1.0 / r_[d] / r_[d];
        }

        I0_ = ( 2.0 * db.source.absorption ) /
              ( M_PI * Kokkos::sqrt( M_PI ) * r_[0] * r_[1] * r_[2] );

        // cut off for 3 standard deviations from heat source center
        w_max_ = Kokkos::log( 3 ) + 2 * Kokkos::log( 10 );
    }

    // Function for temperature solve: forward time-centered space (FTCS) method
    template <class ExecSpace, class IndexSpaceType>
    void solve( ExecSpace exec_space, IndexSpaceType owned_space, ViewType& T,
                ViewType& T0, const double beam_power,
                const double beam_pos[3] )
    {
        // Update temperature views and beam parameters for current time step
        T_ = T;

        T0_ = T0;

        power_ = beam_power;

        for ( std::size_t d = 0; d < 3; ++d )
        {
            position_[d] = beam_pos[d];
        }

        // Tagged versions of temperature solver for architecture optimization
        using memory_space = typename ViewType::memory_space;

        if constexpr ( std::is_same<memory_space, Kokkos::HostSpace>::value )
        {
            Cabana::Grid::grid_parallel_for( "solve", exec_space, owned_space,
                                             HostTag{}, *this );
        }
        else
        {
            Cabana::Grid::grid_parallel_for( "solve", exec_space, owned_space,
                                             DeviceTag{}, *this );
        }
    }

    // Host tagged version of the temperature solver
    KOKKOS_INLINE_FUNCTION
    void operator()( HostTag tag, const int i, const int j, const int k ) const
    {
        // First nearest neighbor stencil for cell at i,j,k: temperature and
        // thermal conductivity
        const double temp_local = T0_( i, j, k, 0 );
        const double temp_px = T0_( i + 1, j, k, 0 );
        const double temp_nx = T0_( i - 1, j, k, 0 );
        const double temp_py = T0_( i, j + 1, k, 0 );
        const double temp_ny = T0_( i, j - 1, k, 0 );
        const double temp_pz = T0_( i, j, k + 1, 0 );
        const double temp_nz = T0_( i, j, k - 1, 0 );

        // Liquid fraction
        const double liquid_fraction =
            Kokkos::fmax( 1.0, Kokkos::fmin( 0.0, ( temp_local - solidus_ ) *
                                                      inv_freezing_range_ ) );
        const double kappa_local =
            kappa_of_temperature( temp_local, liquid_fraction );
        const double kappa_px =
            kappa_of_temperature( temp_px, liquid_fraction );
        const double kappa_nx =
            kappa_of_temperature( temp_nx, liquid_fraction );
        const double kappa_py =
            kappa_of_temperature( temp_py, liquid_fraction );
        const double kappa_ny =
            kappa_of_temperature( temp_ny, liquid_fraction );
        const double kappa_pz =
            kappa_of_temperature( temp_pz, liquid_fraction );
        const double kappa_nz =
            kappa_of_temperature( temp_nz, liquid_fraction );

        double dt_by_rho_cp =
            ( liquid_fraction >= 0.0 && liquid_fraction <= 1.0 )
                ? dt_ / ( rho_cp_ + rho_Lf_by_dT_ )
                : dt_ / ( rho_cp_ );

        const double laplacian_x = laplacian_k(
            temp_local, temp_px, temp_nx, kappa_local, kappa_px, kappa_nx );
        const double laplacian_y = laplacian_k(
            temp_local, temp_py, temp_ny, kappa_local, kappa_py, kappa_ny );
        const double laplacian_z = laplacian_k(
            temp_local, temp_pz, temp_nz, kappa_local, kappa_pz, kappa_nz );

        const double rhs =
            ( laplacian_x + laplacian_y + laplacian_z ) * inv_dx2_ +
            source( tag, i, j, k );

        T_( i, j, k, 0 ) = temp_local + rhs * dt_by_rho_cp;
    }

    // Device tagged version of the temperature solver
    KOKKOS_INLINE_FUNCTION
    void operator()( DeviceTag tag, const int i, const int j,
                     const int k ) const
    {
        // First nearest neighbor stencil for cell at i,j,k: temperature and
        // thermal conductivity
        const double temp_local = T0_( i, j, k, 0 );
        const double temp_px = T0_( i + 1, j, k, 0 );
        const double temp_nx = T0_( i - 1, j, k, 0 );
        const double temp_py = T0_( i, j + 1, k, 0 );
        const double temp_ny = T0_( i, j - 1, k, 0 );
        const double temp_pz = T0_( i, j, k + 1, 0 );
        const double temp_nz = T0_( i, j, k - 1, 0 );

        // Liquid fraction
        const double liquid_fraction =
            Kokkos::fmax( 1.0, Kokkos::fmin( 0.0, ( temp_local - solidus_ ) *
                                                      inv_freezing_range_ ) );
        const double kappa_local =
            kappa_of_temperature( temp_local, liquid_fraction );
        const double kappa_px =
            kappa_of_temperature( temp_px, liquid_fraction );
        const double kappa_nx =
            kappa_of_temperature( temp_nx, liquid_fraction );
        const double kappa_py =
            kappa_of_temperature( temp_py, liquid_fraction );
        const double kappa_ny =
            kappa_of_temperature( temp_ny, liquid_fraction );
        const double kappa_pz =
            kappa_of_temperature( temp_pz, liquid_fraction );
        const double kappa_nz =
            kappa_of_temperature( temp_nz, liquid_fraction );

        double dt_by_rho_cp =
            dt_ / ( rho_cp_ + ( liquid_fraction >= 0.0 ) *
                                  ( liquid_fraction <= 1.0 ) * rho_Lf_by_dT_ );

        const double laplacian_x = laplacian_k(
            temp_local, temp_px, temp_nx, kappa_local, kappa_px, kappa_nx );
        const double laplacian_y = laplacian_k(
            temp_local, temp_py, temp_ny, kappa_local, kappa_py, kappa_ny );
        const double laplacian_z = laplacian_k(
            temp_local, temp_pz, temp_nz, kappa_local, kappa_pz, kappa_nz );

        const double rhs =
            ( laplacian_x + laplacian_y + laplacian_z ) * inv_dx2_ +
            source( tag, i, j, k );

        T_( i, j, k, 0 ) = temp_local + rhs * dt_by_rho_cp;
    }

    // Get temperature-dependent thermal conductivity, using liquid and solid
    // values. Conductivity is capped at the value where temp_local = temp_max_
    KOKKOS_INLINE_FUNCTION
    auto kappa_of_temperature( const double temp_local,
                               const double liquid_fraction ) const
    {
        return ( 1.0 - liquid_fraction ) *
                   ( k_solid_0_ + k_solid_T_ * temp_local ) +
               liquid_fraction *
                   ( k_liquid_0_ +
                     k_liquid_T_ * Kokkos::fmin( temp_local, temp_max_ ) );
    }

    // Get harmonic average of two values
    KOKKOS_INLINE_FUNCTION
    auto harmonic_mean( const double x1, const double x2 ) const
    {
        return 2.0 * x1 * x2 / ( x1 + x2 );
    }

    // First-order centered space laplacian stencil for one direction -
    // temperature-dependent thermal conductivity
    KOKKOS_INLINE_FUNCTION
    auto laplacian_k( const double temp_local, const double temp_positive,
                      const double temp_negative, const double kappa_local,
                      const double kappa_positive,
                      const double kappa_negative ) const
    {
        return ( harmonic_mean( kappa_local, kappa_positive ) *
                     ( temp_positive - temp_local ) -
                 harmonic_mean( kappa_local, kappa_negative ) *
                     ( temp_local - temp_negative ) );
    }

    // Normalized weight for the gaussian source term: x in exp(-x)
    KOKKOS_INLINE_FUNCTION
    auto weight( const int i, const int j, const int k ) const
    {
        double grid_loc[3];
        double dist_to_beam[3];
        int idx[3] = { i, j, k };

        local_mesh_.coordinates( EntityType(), idx, grid_loc );

        dist_to_beam[0] = grid_loc[0] - position_[0];
        dist_to_beam[1] = grid_loc[1] - position_[1];
        dist_to_beam[2] = grid_loc[2] - position_[2];

        return ( dist_to_beam[0] * dist_to_beam[0] * A_inv_[0] ) +
               ( dist_to_beam[1] * dist_to_beam[1] * A_inv_[1] ) +
               ( dist_to_beam[2] * dist_to_beam[2] * A_inv_[2] );
    }

    // Heating source term, device overload.
    KOKKOS_INLINE_FUNCTION
    auto source( DeviceTag, const int i, const int j, const int k ) const
    {
        return I0_ * power_ * Kokkos::exp( -weight( i, j, k ) );
    }

    // Heating source term, host overload.
    KOKKOS_INLINE_FUNCTION
    auto source( HostTag, const int i, const int j, const int k ) const
    {
        // performance improvements on host: scoping the exponential
        if ( power_ )
        {
            double w = weight( i, j, k );

            if ( w < w_max_ )
            {
                return I0_ * power_ * Kokkos::exp( -w );
            }
            else
            {
                return 0.0;
            }
        }
        else
        {
            return 0.0;
        }
    }
};

// Create a solver based on the grid details and simulation inputs.
template <typename MemorySpace>
auto createSolver( Inputs db, Grid<MemorySpace> grid )
{
    using entity_type = typename Grid<MemorySpace>::entity_type;
    using view_type = typename Grid<MemorySpace>::view_type;
    using mesh_type = typename Grid<MemorySpace>::local_mesh_type;

    auto local_mesh = grid.getLocalMesh();

    return Solver<view_type, entity_type, mesh_type>( db, local_mesh );
}

} // namespace Finch

#endif
