#!/bin/sh
#PBS -l select=1:ngpus=1
#PBS -q gpuq_cuda10
#PBS -N CuDNNLSTM

cd $PBS_O_WORKDIR #カレントディレクトリに移動
 
. ~/anaconda3/etc/profile.d/conda.sh #conda動かすためのsシェルスクリプトを実行(?)
conda activate gpu #パッケージの入った仮想環境を起動
python CuLSTM.py　#ここのファイルを任意のものに書き換える
