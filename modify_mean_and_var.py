from sys import stderr
import numpy as np
from os.path import join
from psbody.mesh import Mesh
from fitting.landmarks import load_embedding, landmark_error_3d, mesh_points_by_barycentric_coordinates, load_picked_points
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir, get_unit_factor
import open3d as o3d
import argparse, os
from tqdm import tqdm

def get_config():
    parser = argparse.ArgumentParser(description='modify mean and std and orientation')
    parser.add_argument("--scans", type=str, default= "mesh", help='path of the scan') # for a mesh path, replace 'mesh' to 'lmk' get its corresponding lmk path
    parser.add_argument("--lmks", type=str, default= "lmk", help='path of the output')
    parser.add_argument("--save", type=str, default= "lx_result", help='path of the output')
    args = parser.parse_args()
    return args


def get_mean_std(filename):
    mesh = Mesh(filename=filename)
    mesh.v *= [1, -1, -1] # 沿x轴旋转
    if hasattr(mesh, 'f'):
        mesh.f = modify_face(mesh.f) # TODO: 尚未确定是否需要扭转面片方向
    mean = np.mean(mesh.v, axis=0)
    std = np.std(mesh.v)
    return mean, std, mesh


def modify_face(face):
    return face


def get_lmk(lmk_path):
    if lmk_path.endswith('.npy'):
        lmk = np.load(lmk_path)
    elif lmk_path.endswith('.pp'):
        lmk = load_picked_points(lmk_path)
    return lmk


if __name__ == '__main__':
    eg = './data/scan.obj'
    eg_mean, eg_std, eg_mesh = get_mean_std(eg)
    args = get_config()
    save_root = join('data', args.save)
    os.makedirs(save_root, exist_ok=True)
    save_scan = join(save_root, args.scans)
    os.makedirs(save_scan, exist_ok=True)
    save_lmk = join(save_root, args.lmks)
    os.makedirs(save_lmk, exist_ok=True)
    scans = join('./data/new_cap', args.scans)
    for r, ds, fs in os.walk(scans):
        for f in tqdm(fs):
            if f.endswith("obj"): 
                scan_path = os.path.join(r,f)
                print(scan_path)
                output = join(save_scan, f)
                mean, std, mesh = get_mean_std(scan_path)
                moved_v = (mesh.v - mean) # 把自己的mesh移到原点并归一化
                avg_v = np.mean(moved_v, axis=0)
                eg_v = (eg_mesh.v - eg_mean) # 把参考mesh移到原点并归一化
                avg_eg_v = np.mean(eg_v, axis=0)
                print(f'my origin scan mean: {mean}, origin example mean: {eg_mean}')
                print(f'my scan mean: {np.mean(moved_v, axis=0)}, example mean: {np.mean(eg_v, axis=0)}')
                avg_scale = np.mean(avg_eg_v/avg_v) * 8.5
                print("scale times: ", avg_scale)
                scaled_v = moved_v * avg_scale #  这时的mesh应该和示例大小差不多
                v = moved_v + eg_mean # 没有放大，只是移动了位置
                print(f"my new mean: {np.mean(v, axis=0)}, eg_mean: {eg_mean}")
                write_simple_obj(v, mesh.f if hasattr(mesh, 'f') else None, output)

                # 对应修改关键点坐标
                lmk_path = scan_path.replace(args.scans, args.lmks).replace('obj', 'npy')
                ori_lmk = np.load(lmk_path)
                ori_lmk *= [1, -1, -1]
                lmk_output = join(save_lmk, f.replace('obj', 'npy'))
                moved_lmk = (ori_lmk - mean)
                scaled_lmk = moved_lmk * avg_scale
                modified_lmk = moved_lmk + eg_mean
                np.save(lmk_output, modified_lmk)

                # res_lmk = o3d.geometry.PointCloud()
                # res_lmk.points = o3d.utility.Vector3dVector(modified_lmk)
                # res_mesh = o3d.io.read_triangle_mesh(output)
                # o3d.visualization.draw_geometries([res_mesh, res_lmk, eg_mesh])

                


