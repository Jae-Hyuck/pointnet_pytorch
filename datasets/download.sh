#!/bin/bash

cd $(dirname "$0")
if [ ! -d "data" ]; then
        mkdir data
fi
cd data

# Download dataset for point cloud classification
if [ ! -d "modelnet40_ply_hdf5_2048" ]; then
        wget -c https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
        unzip modelnet40_ply_hdf5_2048.zip
        rm modelnet40_ply_hdf5_2048.zip
fi

# Download HDF5 for indoor 3d semantic segmentation (around 1.6GB)
if [ ! -d "indoor3d_sem_seg_hdf5_data" ]; then
        wget -c https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip
        unzip indoor3d_sem_seg_hdf5_data.zip
        rm indoor3d_sem_seg_hdf5_data.zip
fi

# Download original ShapeNetPart dataset (around 1GB)
if [ ! -d "shapenetcore_partanno_v0" ]; then
        wget -c https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip
        unzip shapenetcore_partanno_v0.zip
        mv PartAnnotation shapenetcore_partanno_v0
        rm shapenetcore_partanno_v0.zip
fi

# Download HDF5 for ShapeNet Part segmentation (around 346MB)
if [ ! -d "shapenet_part_seg_hdf5_data" ]; then
        wget -c https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip
        unzip shapenet_part_seg_hdf5_data.zip
        mv hdf5_data shapenet_part_seg_hdf5_data
        rm shapenet_part_seg_hdf5_data.zip
fi
