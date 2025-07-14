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
  \file SolidificationData.hpp
  \brief Class to output solidification information (e.g. for later
  microstructure) simulation
*/

#ifndef SolidificationData_H
#define SolidificationData_H

#include <array>
#include <chrono>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <Stork_Core.hpp>

#include <Finch_Grid.hpp>
#include <Finch_Inputs.hpp>

namespace Finch
{

template <typename MemorySpace>
class SolidificationData
{
    using memory_space = MemorySpace;
    using exec_space = typename memory_space::execution_space;
    using view_int = Kokkos::View<int*, memory_space>;
    using view_double2D = Kokkos::View<double**, memory_space>;
    using view_int4D = Kokkos::View<int****, memory_space>;
    using view_type_coupled =
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    using DualSRDF = Stork::Structs::SRDF_Dual<double>;
    using DualRDF = Stork::Structs::RDF_Dual<double>;

  private:
    // Needed for file output
    int mpi_rank_;
    std::string folder_name_;
    double liquidus_;
    double dt_;
    double cell_size_;
    bool enabled_;
    std::string format_;
    int fine_factor_;
    view_int count;

    int capacity;
    double x_max, y_max, z_max;
    int nx_solidification, ny_solidification, nz_solidification;

    view_int cellnum;
    view_double2D timesview, thermalsview;
    //    view_int4D tm_view;

  public:
    // Default constructor
    SolidificationData() {}
    // constructor
    SolidificationData( const Inputs& inputs, Grid<memory_space>& grid )
        : mpi_rank_( grid.comm_rank )
        , folder_name_( inputs.sampling.directory_name )
        , liquidus_( inputs.properties.liquidus )
        , dt_( inputs.time.time_step )
        , cell_size_( inputs.space.cell_size )
        , enabled_( inputs.sampling.enabled )
        , format_( inputs.sampling.format )
        , fine_factor_( inputs.sampling.fine_factor )
    {
        count = view_int( "count", 1 );

        capacity = round( grid.getIndexSpace().size() );
        cellnum = view_int(
            Kokkos::ViewAllocateWithoutInitializing( "cellnum" ), capacity );
        timesview = view_double2D(
            Kokkos::ViewAllocateWithoutInitializing( "times" ), capacity, 2 );
        thermalsview = view_double2D(
            Kokkos::ViewAllocateWithoutInitializing( "thermals" ), capacity,
            16 );

        auto local_mesh = grid.getLocalMesh();

        // CA grid - store node data in halo regions in positive x,y,z, but if
        // at a global domain boundary, do not store boundary node data
        x_max = inputs.space.global_high_corner[0];
        y_max = inputs.space.global_high_corner[1];
        z_max = inputs.space.global_high_corner[2];

        if ( std::abs( local_mesh.highCorner( Cabana::Grid::Own(), 0 ) -
                       x_max ) < 1e-10 )
            nx_solidification = grid.num_points_x;
        else
            nx_solidification = grid.num_points_x + 1;
        if ( std::abs( local_mesh.highCorner( Cabana::Grid::Own(), 1 ) -
                       y_max ) < 1e-10 )
            ny_solidification = grid.num_points_y;
        else
            ny_solidification = grid.num_points_y + 1;
        if ( std::abs( local_mesh.highCorner( Cabana::Grid::Own(), 2 ) -
                       z_max ) < 1e-10 )
            nz_solidification = grid.num_points_z;
        else
            nz_solidification = grid.num_points_z + 1;
        //        auto layout =
        //            Cabana::Grid::createArrayLayout( local_grid, 1,
        //            entity_type() );
        //        auto tm =
        //            Cabana::Grid::createArray<int, memory_space>( "tm", layout
        //            );
        //        tm_view = tm->view();
        //        Kokkos::deep_copy(tm_view, 0);
    }

    void updateEvents( Grid<memory_space>& grid, const double time )
    {
        // get local copies from grid
        auto T = grid.getTemperature();
        auto T0 = grid.getPreviousTemperature();
        auto local_mesh = grid.getLocalMesh();
        using entity_type = typename Grid<memory_space>::entity_type;
        double x_max_ = x_max;
        double y_max_ = y_max;
        double z_max_ = z_max;
        double dt = dt_;
        int capacity_ = capacity;
        int ny_solidification_ = ny_solidification;
        int nz_solidification_ = nz_solidification;
        Cabana::Grid::grid_parallel_for(
            "local_grid_for", exec_space(), grid.getIndexSpace(),
            KOKKOS_CLASS_LAMBDA( const int i, const int j, const int k ) {
                double pt[3];
                int idx[3] = { i, j, k };
                local_mesh.coordinates( entity_type(), idx, pt );
                // Loop over owned points, except for global x,y,z bound
                if ( ( pt[0] < x_max_ ) && ( pt[1] < y_max_ ) &&
                     ( pt[2] < z_max_ ) )
                {
                    //                    printf("i %d j %d k %d; x %f y %f z
                    //                    %f\n",i,j,k,pt[0],pt[1],pt[2]);
                    // Count number of vertices above the liquidus on this time
                    // step
                    int vert_above_liquidus = 0;
                    int vert_above_liquidus_old = 0;
                    for ( int n_index = 0; n_index < 8; ++n_index )
                    {
                        const int neighbor_zn = k + ( n_index & 1 );
                        const int neighbor_yn = j + ( ( n_index >> 1 ) & 1 );
                        const int neighbor_xn = i + ( ( n_index >> 2 ) & 1 );

                        vert_above_liquidus_old +=
                            T0( neighbor_xn, neighbor_yn, neighbor_zn, 0 ) >=
                            liquidus_;
                        vert_above_liquidus += T( neighbor_xn, neighbor_yn,
                                                  neighbor_zn, 0 ) >= liquidus_;
                    }

                    //                    if (T(i, j, k, 0) >= liquidus_)
                    //                        tm_view(i, j, k, 0) = 1;
                    // store previous, current temperature state if:
                    // - between 1 and 7 of the vertices were above the liquidus
                    // on either the previous or current time step
                    // - all vertices were above the liquidus and now all
                    // vertices are below the liquidus
                    // - all vertices were below the liquidus and now all
                    // vertices are above the liquidus
                    if ( ( vert_above_liquidus_old % 8 ) ||
                         ( vert_above_liquidus % 8 ) ||
                         ( vert_above_liquidus != vert_above_liquidus_old ) )
                    {
                        auto counter =
                            Kokkos::atomic_fetch_add( &count( 0 ), 1 );
                        //                        printf("k = %d, Z =
                        //                        %f\n",k,pt[2]);
                        // Store 1D index of cell in a way consistent with Stork
                        // - if there's space in the structs
                        if ( counter < capacity_ )
                        {
                            cellnum( counter ) =
                                ( i - 1 ) * ny_solidification_ *
                                    nz_solidification_ +
                                ( j - 1 ) * nz_solidification_ + ( k - 1 );

                            // Store previous, current times
                            timesview( counter, 0 ) = time - dt;
                            timesview( counter, 1 ) = time;

                            // Store vertex temperatures at previous, current
                            // times
                            for ( int n_index = 0; n_index < 8; ++n_index )
                            {
                                const int neighbor_zn = k + ( n_index & 1 );
                                const int neighbor_yn =
                                    j + ( ( n_index >> 1 ) & 1 );
                                const int neighbor_xn =
                                    i + ( ( n_index >> 2 ) & 1 );
                                thermalsview( counter, n_index ) = T0(
                                    neighbor_xn, neighbor_yn, neighbor_zn, 0 );
                                thermalsview( counter, n_index + 8 ) = T(
                                    neighbor_xn, neighbor_yn, neighbor_zn, 0 );
                            }
                        }
                    }
                }
            } );
    }

    // Update the solidification data
    void update( Grid<memory_space>& grid, const double time )
    {
        if ( !enabled_ )
        {
            return;
        }

        auto count_old_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), count );
        int old_count = count_old_host( 0 );
        updateEvents( grid, time );

        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), count );

        int new_count = count_host( 0 );

        // more events were added than the current view capacity.
        // resize view and update events starting from the previous counter.
        if ( new_count >= capacity )
        {
            capacity = 2.0 * new_count;
            Kokkos::resize( Kokkos::WithoutInitializing, cellnum, capacity );
            Kokkos::resize( Kokkos::WithoutInitializing, timesview, capacity,
                            2 );
            Kokkos::resize( Kokkos::WithoutInitializing, thermalsview, capacity,
                            16 );

            Kokkos::deep_copy( count, old_count );

            updateEvents( grid, time );
        }

        // view size is within 90% of capacity. double current size.
        else if ( new_count / capacity > 0.9 )
        {
            capacity = 2.0 * new_count;

            Kokkos::resize( Kokkos::WithoutInitializing, cellnum, capacity );
            Kokkos::resize( Kokkos::WithoutInitializing, timesview, capacity,
                            2 );
            Kokkos::resize( Kokkos::WithoutInitializing, thermalsview, capacity,
                            16 );
        }
    }

    // Return all data for the events that have been recorded during the
    // simulation
    auto get()
    {
        //        auto events_host =
        //            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
        //            events );
        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), count );
        //        // Resize the host copy so only valid events get copied.
        //        Kokkos::resize( events_host, count_host( 0 ), nCmpts );
        //        // Create a View on the host with fixed layout for coupling.
        view_type_coupled copied_data(
            Kokkos::ViewAllocateWithoutInitializing( "copied_data" ),
            count_host( 0 ), 1 );
        //        Kokkos::deep_copy( copied_data, events_host );
        return copied_data;
    }

    // Write the solidification data to separate files for each MPI rank
    void write( Grid<memory_space>& grid, MPI_Comm comm )
    {
        if ( !enabled_ )
        {
            return;
        }

        std::chrono::high_resolution_clock::time_point
            start_solidification_print_time =
                std::chrono::high_resolution_clock::now();

        auto local_mesh = grid.getLocalMesh();
        //        int sum_tm = 0;
        //        auto T = grid.getTemperature();

        //        Cabana::Grid::grid_parallel_reduce(
        //            "local_grid_for", exec_space(), grid.getIndexSpace(),
        //            KOKKOS_CLASS_LAMBDA( const int i, const int j, const int
        //            k, int &update ) {
        //                if (tm_view(i, j, k, 0) == 1)
        //                    update++;
        //        }, sum_tm);
        //        std::cout << "Points that went above the liquidus: " << sum_tm
        //        << std::endl;
        // Init SRDF object
        DualSRDF SRDF;
        Stork::Structs::RegularGrid_Header<double, host_space> header =
            SRDF.host_header;

        // Origin point
        header.global_x0() = local_mesh.lowCorner( Cabana::Grid::Own(), 0 );
        header.global_y0() = local_mesh.lowCorner( Cabana::Grid::Own(), 1 );
        header.global_z0() = local_mesh.lowCorner( Cabana::Grid::Own(), 2 );
        std::cout << "Rank " << mpi_rank_ << " low corner is at "
                  << header.global_x0() << ", " << header.global_y0() << ", "
                  << header.global_z0() << std::endl;

        // Each MPI rank interpolates only on the local grid - no awareness of
        // global grid needed
        header.global_i0() = 0;
        header.global_j0() = 0;
        header.global_k0() = 0;

        // Number of points in each direction - includes halo in positive
        // directions unless at the global domain bound
        header.local_inum() = nx_solidification;
        header.local_jnum() = ny_solidification;
        header.local_knum() = nz_solidification;

        // Cell size
        header.gridResolution() = cell_size_;

        // Liquidus temperature
        SRDF.T_critical = liquidus_;
        // Resize views based on final count of temperature snapshots
        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), count );
        Kokkos::resize( cellnum, count_host( 0 ) );
        Kokkos::resize( timesview, count_host( 0 ), 2 );
        Kokkos::resize( thermalsview, count_host( 0 ), 16 );
        // Number of snapshots
        SRDF.numSnaps = count_host( 0 );
        std::cout << "Rank " << mpi_rank_
                  << " number of snapshots: " << count_host( 0 ) << std::endl;
        // Copy views to host
        auto cellnum_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), cellnum );
        auto timesview_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), timesview );
        auto thermalsview_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), thermalsview );

        // Create tuples to order by cell number
        std::vector<int> old_list_position( count_host( 0 ) );
        for ( int n = 0; n < count_host( 0 ); n++ )
            old_list_position[n] = n;
        std::vector<std::tuple<int, int>> snapshots;
        snapshots.reserve( count_host( 0 ) );

        //        std::sort(unique_verts.begin(), unique_verts.end());
        //        std::vector<int>::iterator it;
        //        it = std::unique(unique_verts.begin(), unique_verts.end());
        //        unique_verts.resize(std::distance(unique_verts.begin(), it));
        //        std::cout << "Number of unique vertices: " <<
        //        unique_verts.size() << std::endl;

        for ( int n = 0; n < count_host( 0 ); n++ )
        {
            snapshots.push_back(
                std::make_tuple( cellnum_host( n ), old_list_position[n] ) );
        }
        // Sorting from low to high
        std::sort( snapshots.begin(), snapshots.end() );

        // Create empty SRDF views based on count
        SRDF.template Make_Data_Views<device_space>( count_host( 0 ) );
        // Get reference to SRDF views on host
        //        Stork::Structs::SRDF_Data<double, host_space>& data =
        //        SRDF.host_data;

        // Fill SRDF views from sorted Finch data
        for ( int n = 0; n < count_host( 0 ); n++ )
        {
            SRDF.host_data.cellNum_view( n ) = std::get<0>( snapshots[n] );
            //            int k = std::get<0>(snapshots[n]) % grid.num_points_z;
            //            if (k < 15)
            //                std::cout << "k = " << k << std::endl;
            const int old_list_pos = std::get<1>( snapshots[n] );
            SRDF.host_data.times_view( 2 * n ) =
                timesview_host( old_list_pos, 0 );
            SRDF.host_data.times_view( 2 * n + 1 ) =
                timesview_host( old_list_pos, 1 );
            for ( int vert = 0; vert < 16; vert++ )
                SRDF.host_data.thermals_view( 16 * n + vert ) =
                    thermalsview_host( old_list_pos, vert );
        }

        //        int last_cell = -1;
        //        bool melted_yn = false;
        //        bool solidified_yn = false;
        //        for (int n = 0; n < count_host(0); n++) {
        //            int cell_num = SRDF.host_data.cellNum_view( n );
        //            if ((cell_num != last_cell) && (melted_yn !=
        //            solidified_yn)) {
        //                std::cout << "Cell " << cell_num << " did something
        //                weird" << std::endl; melted_yn = false; solidified_yn
        //                = false; last_cell = cell_num;
        //            }
        //            double old_temp = SRDF.host_data.thermals_view( 16 * n );
        //            double new_temp = SRDF.host_data.thermals_view( 16 * n  +
        //            8); if ((old_temp < SRDF.T_critical) && (new_temp >=
        //            SRDF.T_critical))
        //                melted_yn = true;
        //            if ((old_temp >= SRDF.T_critical) && (new_temp <
        //            SRDF.T_critical))
        //                solidified_yn = true;
        //            if (cell_num == 11569) {
        //                std::cout << "Cell 11569 Temps " << old_temp << ", "
        //                << new_temp << std::endl;
        //            }
        //            if (cell_num == 11570) {
        //                std::cout << "Cell 11570 Temps " << old_temp << ", "
        //                << new_temp << std::endl;
        //            }
        //        }
        //        for (int n = 0; n < count_host(0); n++) {
        //            std::cout << "Cell num " << SRDF.host_data.cellNum_view( n
        //            ) << std::endl; std::cout << "Time 0 " <<
        //            SRDF.host_data.times_view( 2 * n ) << std::endl; std::cout
        //            << "Time 1 " << SRDF.host_data.times_view( 2 * n + 1 ) <<
        //            std::endl; for (int vert = 0; vert < 16; vert++)
        //                std::cout << "Temp vert " << vert << " " <<
        //                SRDF.host_data.thermals_view( 16 * n + vert ) <<
        //                std::endl;
        //        }
        // Fine Interpolation
        DualRDF RDF = Stork::Run::Interpolate_SRDF_to_RDF<double, host_space,
                                                          double, device_space>(
            SRDF, fine_factor_ );

        // Copy data from device back to host
        RDF.template Make_Data_Mirrors<device_space, host_space>();
        RDF.template Copy_All<device_space, host_space>();
        //        std::cout << "Number of melting/solidification events: " <<
        //        RDF.numEvents << std::endl;
        std::cout << "Rank " << mpi_rank_ << " new domain "
                  << RDF.host_header.local_inum() << ","
                  << RDF.host_header.local_jnum() << ","
                  << RDF.host_header.local_knum() << std::endl;

        // create directory is not present, otherwise overwrite existing files
        if ( mkdir( folder_name_.c_str(), 0777 ) != -1 )
        {
            std::cout << "Creating directory: " << folder_name_ << std::endl;
        }

        std::string filename( folder_name_ + "/data_" +
                              std::to_string( mpi_rank_ ) );

        // Output Data to File
        Stork::IO::Output_RDF_csv<double>( RDF, filename );
        MPI_Barrier( comm );
        std::chrono::high_resolution_clock::time_point
            end_solidification_print_time =
                std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds =
            end_solidification_print_time - start_solidification_print_time;
        if ( mpi_rank_ == 0 )
            std::cout << "Solidification data written in " << std::fixed
                      << std::setprecision( 6 ) << elapsed_seconds.count()
                      << " seconds" << std::endl;
        //        for (int n_index=0; n_index < 8; ++n_index)
        //        {
        //            const int neighbor_xn = (n_index & 1);
        //            const int neighbor_yn = ((n_index >> 1) & 1);
        //            const int neighbor_zn = ((n_index >> 2) & 1);
        //            std::cout << "index " << n_index << " i,j,k" <<
        //            neighbor_xn << "," << neighbor_yn << "," << neighbor_zn <<
        //            std::endl;
        //        }
    }
};

} // namespace Finch

#endif
