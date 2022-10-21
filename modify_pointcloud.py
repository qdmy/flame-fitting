import numpy as np
from os.path import join
from psbody.mesh import Mesh
from fitting.landmarks import load_embedding, landmark_error_3d, mesh_points_by_barycentric_coordinates, load_picked_points
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir, get_unit_factor
import open3d as o3d
import argparse, os
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

def get_config():
    parser = argparse.ArgumentParser(description='modify mean and std and orientation')
    parser.add_argument("--scans", type=str, default= "mesh", help='path of the scan') # for a mesh path, replace 'mesh' to 'lmk' get its corresponding lmk path
    parser.add_argument("--lmks", type=str, default= "lmk", help='path of the output')
    parser.add_argument("--save", type=str, default= "lx_result", help='path of the output')
    args = parser.parse_args()
    return args


def x_rotate(v):
    return v*[1, -1, -1]

def transl(v, old_mean, new_mean):
    return v-old_mean+new_mean

def transl_scale(v, old_mean, old_std, new_mean, new_std):
    return (v-old_mean)/old_std*new_std+new_mean
    
def modify_face(face):
    return face

def get_vertice_mean_std(v):
    return np.mean(v, axis=0), np.std(v)

def get_mean_std(filename):
    mesh = Mesh(filename=filename)
    if hasattr(mesh, 'f'):
        mesh.f = modify_face(mesh.f) # TODO: 尚未确定是否需要扭转面片方向
    mean = np.mean(mesh.v, axis=0)
    std = np.std(mesh.v)
    return mean, std, mesh

def flamefit_test():
    eg = './data/scan.obj'
    lmk = './data/scan_lmks.npy'
    eg_mean, eg_std, eg_mesh = get_mean_std(eg) # mean x-y-z分开算, std整体算
    eg_lmk = np.load(lmk)
    print(f'my example scan mean: {eg_mean}, std: {eg_std}')

    my_scan = "/mnt/cephfs/home/liuxu/cvte/tools/flame-fitting/data/test/mesh/3_pointcloud.obj"
    my_lmk = "/mnt/cephfs/home/liuxu/cvte/tools/flame-fitting/data/test/lmk/3_pointcloud.npy"
    mean, std, mesh = get_mean_std(my_scan)
    lmk = np.load(my_lmk)
    v = mesh.v
    print(f'my origina scan mean: {mean}, std: {std}')

    v = x_rotate(v)
    lmk = x_rotate(lmk)
    write_simple_obj(v, mesh.f if hasattr(mesh, 'f') else None, my_scan.replace('.obj', '_x.obj'))
    np.save(my_lmk.replace('.npy', '_x.npy'), lmk)
    mean, std = get_vertice_mean_std(v)
    print(f'my rotated scan mean: {mean}, std: {std}')

    v_transl = transl(v, mean, eg_mean)
    lmk_transl = transl(lmk, mean, eg_mean)
    write_simple_obj(v_transl, mesh.f if hasattr(mesh, 'f') else None, my_scan.replace('.obj', '_x_transl.obj'))
    np.save(my_lmk.replace('.npy', '_x_transl.npy'), lmk_transl)
    mean_transl, std_transl = get_vertice_mean_std(v_transl)
    print(f'my transla scan mean: {mean_transl}, std: {std_transl}')
    

    v = transl_scale(v, mean, std, eg_mean, eg_std)
    lmk = transl_scale(lmk, mean, std, eg_mean, eg_std)
    write_simple_obj(v, mesh.f if hasattr(mesh, 'f') else None, my_scan.replace('.obj', '_x_transl_scale.obj'))
    np.save(my_lmk.replace('.npy', '_x_transl_scale.npy'), lmk)
    mean, std = get_vertice_mean_std(v)
    print(f'my tra_sca scan mean: {mean}, std: {std}')


    # scale to similar size based on lmk
    eg_lmk = eg_lmk - eg_mean
    lmk = lmk - mean # 关键点相对于原点的坐标
    times = np.mean(np.mean(eg_lmk/lmk, axis=1)) # 关键点的avg倍数
    v = (v - mean)*times
    lmk = lmk*times
    mean, std = get_vertice_mean_std(v)
    print(f'my fang_da scan mean: {mean}, std: {std}')
    v = transl_scale(v, mean, std, eg_mean, eg_std)
    lmk = transl_scale(lmk, mean, std, eg_mean, eg_std)
    write_simple_obj(v, mesh.f if hasattr(mesh, 'f') else None, my_scan.replace('.obj', '_x_transl_scale_fangda.obj'))
    np.save(my_lmk.replace('.npy', '_x_transl_scale_fangda.npy'), lmk)
    mean, std = get_vertice_mean_std(v)
    print(f'my finally scan mean: {mean}, std: {std}')



# 只需要旋转并平移一下就ok了，调这个函数
def liuxu_flamefit():
    eg = './data/scan.obj'
    lmk = './data/scan_lmks.npy'
    eg_mean, eg_std, eg_mesh = get_mean_std(eg) # mean x-y-z分开算, std整体算
    eg_lmk = np.load(lmk)
    print(f'my example scan mean: {eg_mean}, std: {eg_std}')

    my_scan = "/mnt/cephfs/home/liuxu/cvte/tools/flame-fitting/data/new_cap/mesh/0_face.obj"
    my_lmk = "/mnt/cephfs/home/liuxu/cvte/tools/flame-fitting/data/new_cap/lmk/0_face.npy"
    mean, std, mesh = get_mean_std(my_scan)
    lmk = np.load(my_lmk)[-51:]
    v = mesh.v
    print(f'my origina scan mean: {mean}, std: {std}')

    v = x_rotate(v)
    lmk = x_rotate(lmk)
    # write_simple_obj(v, mesh.f if hasattr(mesh, 'f') else None, my_scan.replace('.obj', '_x.obj'))
    # np.save(my_lmk.replace('.npy', '_x.npy'), lmk)
    mean, std = get_vertice_mean_std(v)
    # print(f'my rotated scan mean: {mean}, std: {std}')

    v_transl = transl(v, mean, eg_mean) # 到这一步得到的obj，fit效果最好
    lmk_transl = transl(lmk, mean, eg_mean)
    write_simple_obj(v_transl, mesh.f if hasattr(mesh, 'f') else None, my_scan.replace('.obj', '_x_transl.obj'))
    np.save(my_lmk.replace('.npy', '_x_transl.npy'), lmk_transl)
    mean_transl, std_transl = get_vertice_mean_std(v_transl)
    print(f'my transla scan mean: {mean_transl}, std: {std_transl}')

    # v = transl_scale(v, mean, std, eg_mean, eg_std)
    # lmk = transl_scale(lmk, mean, std, eg_mean, eg_std)
    # write_simple_obj(v, mesh.f if hasattr(mesh, 'f') else None, my_scan.replace('.obj', '_x_transl_scale.obj'))
    # np.save(my_lmk.replace('.npy', '_x_transl_scale.npy'), lmk)
    # mean, std = get_vertice_mean_std(v)
    # print(f'my tra_sca scan mean: {mean}, std: {std}')


def get_lmk_meanstd(lmk):
    mean = np.mean(lmk, axis=0)
    std = np.std(lmk)
    return mean, std


# 只需要旋转并平移一下就ok了，调这个函数
def liuxu_modify_basedon_lmk():
    eg = 'data/scan.obj'
    lmk = 'data/scan_lmks.npy'
    eg_lmk = np.load(lmk)
    eg_mean, eg_std = get_lmk_meanstd(eg_lmk) # mean x-y-z分开算, std整体算
    print(f'my example lmk mean: {eg_mean}, std: {eg_std}')

    my_scan = "data/lizhenliang2/lizhenliang2_down10.ply"
    my_lmk = "data/lizhenliang2/lizhenliang2_picked_points.pp"
    lmk = get_lmk(my_lmk)[-51:]
    mean, std = get_lmk_meanstd(lmk)
    mesh = Mesh(filename=my_scan)
    v = mesh.v
    print(f'my origina lmk mean: {mean}, std: {std}')

    v = x_rotate(v)
    lmk = x_rotate(lmk)
    mean, std = get_lmk_meanstd(lmk)

    v_transl = transl(v, mean, eg_mean) # 到这一步得到的obj，fit效果最好
    lmk_transl = transl(lmk, mean, eg_mean)
    write_simple_obj(v_transl, mesh.f if hasattr(mesh, 'f') else None, my_scan.replace('.ply', '_x_transl_by_lmk.obj'))
    np.save(my_lmk.replace('.pp', '_x_transl_by_lmk.npy'), lmk_transl)
    mean_transl, std_transl = get_lmk_meanstd(lmk_transl)
    print(f'my transla lmk mean: {mean_transl}, std: {std_transl}')

    # v = transl_scale(v, mean, std, eg_mean, eg_std)
    # lmk = transl_scale(lmk, mean, std, eg_mean, eg_std)
    # write_simple_obj(v, mesh.f if hasattr(mesh, 'f') else None, my_scan.replace('.obj', '_x_transl_scale_by_lmk.obj'))
    # np.save(my_lmk.replace('.npy', '_x_transl_scale_by_lmk.npy'), lmk)
    # mean, std = get_lmk_meanstd(lmk)
    # print(f'my tra_sca lmk mean: {mean}, std: {std}')

    # print(f'the 13th lmk of example: {eg_lmk[13]}, my: {lmk[13]}')


def get_lmk(lmk_path):
    if lmk_path.endswith('.npy'):
        lmk = np.load(lmk_path)
    elif lmk_path.endswith('.pp'):
        lmk = load_picked_points(lmk_path)
    return lmk


def stupid_test():
    eg = './data/scan.obj'
    eg_mean, eg_std, eg_mesh = get_mean_std(eg)
    args = get_config()
    save_root = join('data', args.save)
    os.makedirs(save_root, exist_ok=True)
    save_scan = join(save_root, args.scans)
    os.makedirs(save_scan, exist_ok=True)
    save_lmk = join(save_root, args.lmks)
    os.makedirs(save_lmk, exist_ok=True)
    scans = join('./data/test', args.scans)
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

                


# 只需要旋转并平移一下就ok了，调这个函数
def modify(my_scan, my_lmk):
    eg = 'data/scan.obj'
    lmk = 'data/scan_lmks.npy'
    eg_lmk = np.load(lmk)
    eg_mean, eg_std = get_lmk_meanstd(eg_lmk) # mean x-y-z分开算, std整体算
    logger.info(f'my example lmk mean: {eg_mean}, std: {eg_std}')

    lmk = get_lmk(my_lmk)[-51:]
    mean, std = get_lmk_meanstd(lmk)
    mesh = Mesh(filename=my_scan)
    v = mesh.v
    logger.info(f'my origina lmk mean: {mean}, std: {std}')

    v = x_rotate(v)
    lmk = x_rotate(lmk)
    mean, std = get_lmk_meanstd(lmk)

    v_transl = transl(v, mean, eg_mean) # 到这一步得到的obj，fit效果最好
    lmk_transl = transl(lmk, mean, eg_mean)
    write_simple_obj(v_transl, mesh.f if hasattr(mesh, 'f') else None, my_scan.replace('.ply', '_x_transl_by_lmk.obj'))
    np.save(my_lmk.replace('.pp', '_x_transl_by_lmk.npy'), lmk_transl)
    mean_transl, std_transl = get_lmk_meanstd(lmk_transl)
    logger.info(f'my transla lmk mean: {mean_transl}, std: {std_transl}')

    trans = -mean + eg_mean
    logger.info(f"trans: {trans}")

    return my_scan.replace('.ply', '_x_transl_by_lmk.obj'), my_lmk.replace('.pp', '_x_transl_by_lmk.npy'), trans



if __name__ == '__main__':
    # flamefit_test()
    # liuxu_flamefit()
    liuxu_modify_basedon_lmk()
