'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this computer program. 
Using this computer program means that you agree to the terms in the LICENSE file (https://flame.is.tue.mpg.de/modellicense) included 
with the FLAME model. Any use not explicitly granted by the LICENSE is prohibited.

Copyright 2020 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its 
Max Planck Institute for Intelligent Systems. All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''

import numpy as np
import chumpy as ch
import os, json, sys
from os.path import join
from pathlib import Path
import argparse
import logging
from psbody.mesh import Mesh
from smpl_webuser.serialization import load_model
from sbody.mesh_distance import ScanToMesh
from sbody.robustifiers import GMOf
from sbody.alignment.objectives import sample_from_mesh
from fitting.landmarks import load_embedding, landmark_error_3d, mesh_points_by_barycentric_coordinates, load_picked_points
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir, get_unit_factor
from modify_pointcloud import modify, x_rotate, transl
import open3d as o3d
# -----------------------------------------------------------------------------

def init_logging(save_path: Path):
    format = '{asctime:s}|{levelname:8s}|{lineno:4d}|{filename:25s}|{message:s}'
    formatter = logging.Formatter(fmt=format, style='{')
    handlers = logging.root.handlers
    for existed_handler in handlers:
        existed_handler.setFormatter(formatter)
        if isinstance(existed_handler, logging.StreamHandler):
            local_primary = False
    if local_primary:
        console_handler = logging.StreamHandler(stream=sys.stderr)
        handlers.append(console_handler)
    filename = 'experiment.log'
    file_handler = logging.FileHandler(save_path / filename)
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)
    logging.root.handlers = handlers
    logging.root.level = logging.INFO

    return logging.getLogger(__name__)


def compute_approx_scale(lmk_3d, model, lmk_face_idx, lmk_b_coords, opt_options=None):
    """ function: compute approximate scale to align scan and model

    input: 
        lmk_3d: input landmark 3d, in shape (N,3)
        model: FLAME face model
        lmk_face_idx, lmk_b_coords: landmark embedding, in face indices and barycentric coordinates
        opt_options: optimizaton options

    output:
        model.r: fitted result vertices
        model.f: fitted result triangulations (fixed in this code)
        parms: fitted model parameters # 这里就是flame论文里写的，先rigid fit把模型系数确定

    """

    scale = ch.ones(1)
    scan_lmks = scale*ch.array(lmk_3d)
    model_lmks = mesh_points_by_barycentric_coordinates( model, model.f, lmk_face_idx, lmk_b_coords )
    lmk_err = scan_lmks-model_lmks

    # options, 与默认优化选项一致
    if opt_options is None:
        print("fit_lmk3d(): no 'opt_options' provided, use default settings.")
        import scipy.sparse as sp
        opt_options = {}
        opt_options['disp']    = 1
        opt_options['delta_0'] = 0.1
        opt_options['e_3']     = 1e-4
        opt_options['maxiter'] = 2000
        sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
        opt_options['sparse_solver'] = sparse_solver

    # on_step callback
    def on_step(_):
        pass

    ch.minimize( fun      = lmk_err,
                 x0       = [ scale, model.trans, model.pose[:3] ], # model.pose只拿前三个就是因为目前是rigid fit，只需要优化root joint
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    return scale.r

# -----------------------------------------------------------------------------

def fit_scan(  scan,                        # input scan
               lmk_3d,                      # input scan landmarks
               model,                       # model
               lmk_face_idx, lmk_b_coords,  # landmark embedding
               weights,                     # weights for the objectives
               gmo_sigma,                   # weight of the robustifier
               shape_num=300, expr_num=100, opt_options=None ):
    
    """ function: fit FLAME model to a 3D scan

    input: 
        scan: input scan
        lmk_3d: input landmark 3d, in shape (N,3)
        model: FLAME face model
        lmk_face_idx, lmk_b_coords: landmark embedding, in face indices and barycentric coordinates
        weights: weights for each objective
        shape_num, expr_num: numbers of shape and expression compoenents used
        opt_options: optimizaton options

    output:
        model.r: fitted result vertices
        model.f: fitted result triangulations (fixed in this code)
        parms: fitted model parameters

    """

    # variables
    shape_idx      = np.arange( 0, min(300,shape_num) )        # valid shape component range in "betas": 0-299
    expr_idx       = np.arange( 300, 300+min(100,expr_num) )   # valid expression component range in "betas": 300-399
    used_idx       = np.union1d( shape_idx, expr_idx )
    model.betas[:] = np.random.rand( model.betas.size ) * 0.0  # initialized to zero # 初始化所有flame的系数为0
    model.pose[:]  = np.random.rand( model.pose.size ) * 0.0   # initialized to zero
    free_variables = [model.trans, model.pose, model.betas[used_idx]] 
    
    # weights
    logger.info("fit_scan(): use the following weights:")
    for kk in weights.keys():
        logger.info("fit_scan(): weights['%s'] = %f" % ( kk, weights[kk] ))

    # objectives
    # landmark error
    # 根据官方指定的关键点所在面片索引以及重心坐标，得到给定的关键点与flame真实关键点之间每个坐标间的差
    lmk_err = landmark_error_3d(mesh_verts=model, mesh_faces=model.f,  lmk_3d=lmk_3d, lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords)

    # scan-to-mesh distance, measuring the distance between scan vertices and the closest points in the model's surface
    sampler = sample_from_mesh(scan, sample_type='vertices')
    s2m = ScanToMesh(scan, model, model.f, scan_sampler=sampler, rho=lambda x: GMOf(x, sigma=gmo_sigma)) # 这里的 x 在后面就是一个 MeshDistanceSquared 的对象。没看懂。。。

    # regularizer
    shape_err = weights['shape'] * model.betas[shape_idx] 
    expr_err  = weights['expr']  * model.betas[expr_idx] 
    pose_err  = weights['pose']  * model.pose[3:] # exclude global rotation

    objectives = {'s2m': weights['s2m']*s2m, 'lmk': weights['lmk']*lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err} 

    # options
    if opt_options is None:
        print("fit_lmk3d(): no 'opt_options' provided, use default settings.")
        import scipy.sparse as sp
        opt_options = {}
        opt_options['disp']    = 1
        opt_options['delta_0'] = 0.1
        opt_options['e_3']     = 1e-4
        opt_options['maxiter'] = 2000
        sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
        opt_options['sparse_solver'] = sparse_solver

    # on_step callback
    def on_step(_):
        pass
        
    # optimize
    # step 1: rigid alignment
    from time import time
    timer_start = time()
    logger.info("step 1: start rigid fitting...")
    ch.minimize( fun      = lmk_err,
                 x0       = [ model.trans, model.pose[:3] ], # 算出flame generic model需要平移多少(trans)，整体旋转多少(pose[:3])
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    logger.info("step 1: fitting done, in %f sec\n" % ( timer_end - timer_start )) # 先根据关键点，调整平移和整体旋转，把flame移到与输入点云大致对齐的方向和位置

    # step 2: non-rigid alignment
    timer_start = time()
    logger.info("step 2: start non-rigid fitting...")    # 然后开始调整系数，让scan上采样到的点到flame模型上的距离最短
    ch.minimize( fun      = objectives,
                 x0       = free_variables, # 算出为了让objectives最小，flame generic model应该平移多少，pose、shape和expression的系数都是多少
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    logger.info("step 2: fitting done, in %f sec\n" % ( timer_end - timer_start ))

    # return results
    parms = { 'trans': model.trans.r, 'pose': model.pose.r, 'betas': model.betas.r }
    return model.r, model.f, parms


def get_config():
    parser = argparse.ArgumentParser(description='modify mean and std and orientation')
    parser.add_argument("--scan", type=str, default= "mesh", help='path of the scan') # for a mesh path, replace 'mesh' to 'lmk' get its corresponding lmk path
    parser.add_argument("--lmk", type=str, default= "lmk", help='path of the output')
    parser.add_argument("--save", type=str, default= "lx_result", help='path of the output')
    parser.add_argument("--fit_dir", default=False, action="store_true", help='path of the output')
    args = parser.parse_args()
    return args


def run_fitting(scan_path, scan_lmk_path, output_path):
    # # input scan
    # scan_path = "/mnt/cephfs/home/liuxu/cvte/tools/flame-fitting/data/lx_result/mesh/3_pointcloud.obj"

    # # landmarks of the scan
    # scan_lmk_path = "/mnt/cephfs/home/liuxu/cvte/tools/flame-fitting/data/lx_result/lmk/3_pointcloud.npy"

    # measurement unit of landmarks ['m', 'cm', 'mm', 'NA'] 
    # When using option 'NA', the scale of the scan will be estimated by rigidly aligning model and scan landmarks
    scan_unit = 'NA' 

    scan = Mesh(filename=scan_path)
    logger.info(f"loaded scan from: {scan_path}")
    if '.npy' in scan_lmk_path:
        f = np.load
    elif '.pp' in scan_lmk_path:
        f = load_picked_points
    else:
        raise Exception
    lmk_3d = f(scan_lmk_path)
    logger.info(f"loaded scan landmark from: {scan_lmk_path}")

    # model
    model_path = './models/generic_model.pkl' # change to 'female_model.pkl' or 'male_model.pkl', if gender is known
    model = load_model(model_path)       # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    logger.info(f"loaded model from: {model_path}")

    # landmark embedding
    lmk_emb_path = './models/flame_static_embedding.pkl' 
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    logger.info("loaded lmk embedding")

    # scale scans and scan landmarks to be in the same local coordinate systems as the FLAME model
    if scan_unit.lower() == 'na':
        logger.info('No scale specifiec - compute approximate scale based on the landmarks')
        scale_factor = compute_approx_scale(lmk_3d, model, lmk_face_idx, lmk_b_coords)
        logger.info('Scale factor: %f' % scale_factor)
    else:
        scale_factor = get_unit_factor('m') / get_unit_factor(scan_unit)
        logger.info('Scale factor: %f' % scale_factor)        

    # if scale_factor < 0: # 如果是按照一个负值进行scale，那么面片就要转一下，不然mesh的表面就变背面了
    #     print('reverse face...')
    #     ori_f = scan.f
    #     scan.f = np.array([ori_f[:,0], ori_f[:,2], ori_f[:,1]]).T

    # x*1, y*-1, z*-1
    scan.v[:] *= scale_factor#np.array([scale_factor, -scale_factor, -scale_factor]).reshape(1,3)
    lmk_3d[:] *= scale_factor#np.array([scale_factor, -scale_factor, -scale_factor]).reshape(1,3)
    # if scale_factor != 1:
    #     np.save(scan_lmk_path.replace('.npy', '_scaled.npy'), lmk_3d)

    # output
    # output_dir = './output'
    # safe_mkdir(output_dir)

    # # output scaled scan for reference (output scan fit and the scan should be spatially aligned)
    # output_path = join( output_dir, f'{out_name}_scale.obj' )    
    write_simple_obj( mesh_v=scan.v, mesh_f=scan.f if hasattr(scan, 'f') else None, filepath=output_path.replace('.obj', '_scaled.obj'), verbose=False )

    # # 沿z轴旋转，对于模板才是正的
    # # x*1, y*-1, z*-1
    # scan.v[:] *= np.array([-1, 1, -1]).reshape(1,3)
    # lmk_3d[:] *= np.array([-1, 1, -1]).reshape(1,3)

    # weights
    weights = {}
    # scan vertex to model surface distance term
    weights['s2m']   = 2.0   
    # landmark term
    weights['lmk']   = 0 # 1e-2
    # shape regularizer (weight higher to regularize face shape more towards the mean)
    weights['shape'] = 1e-4
    # expression regularizer (weight higher to regularize facial expression more towards the mean)
    weights['expr']  = 1e-4
    # regularization of head rotation around the neck and jaw opening (weight higher for more regularization)
    weights['pose']  = 1e-3
    # Parameter of the Geman-McClure robustifier (higher weight for a larger bassin of attraction which makes it less robust to outliers)
    gmo_sigma = 1e-4

    # optimization options
    import scipy.sparse as sp
    opt_options = {}
    opt_options['disp']    = 1
    opt_options['delta_0'] = 0.1
    opt_options['e_3']     = 1e-4
    opt_options['maxiter'] = 2000
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0] # 用共轭梯度迭代法求解Aw = x 里的w，返回的[0]即w，[1]是融合信息
    opt_options['sparse_solver'] = sparse_solver

    # run fitting
    mesh_v, mesh_f, parms = fit_scan(   scan=scan,                                             # input scan
                                        lmk_3d=lmk_3d,                                         # input landmark 3d
                                        model=model,                                           # model
                                        lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords,  # landmark embedding
                                        weights=weights,                                       # weights for the objectives
                                        gmo_sigma=gmo_sigma,                                   # parameter of the regularizer
                                        shape_num=300, expr_num=100, opt_options=opt_options ) # options

    # write result
    # output_path = join( output_dir, f'{out_name}.obj' )
    write_simple_obj( mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False )
    logger.info(f'fitted output mesh saved to: {output_path}') 

    # 保存scale_factor
    parms["scale_factor"] = scale_factor
    return parms
# -----------------------------------------------------------------------------

def run_fitting_dir(args):
    root = join('./data/new_cap', args.scan)
    for r, ds, fs in os.walk(root):
        for f in fs:
            if f.endswith('obj') and "transl_scale_by_lmk" in f: # 后一个条件来限制测试对象
                scan_path = join(r, f)
                print(scan_path)
                scan_lmk_path = scan_path.replace(args.scan, args.lmk).replace('.obj', '.npy')
                out_name = scan_path.replace('new_cap', args.save)
                run_fitting(scan_path, scan_lmk_path, out_name)


def colmap_fitting(pcd, pp):
    # pointcloud downsample
    pcd_path = pcd.replace(".ply", "_down10.ply")
    pcd = o3d.io.read_point_cloud(pcd)
    downpcd = pcd.uniform_down_sample(10)
    o3d.io.write_point_cloud(pcd_path, downpcd)
    # 1. modify pointcloud
    obj_path, lmk_path, trans = modify(pcd_path, pp)
    # 2. fitting
    out_name = obj_path.replace('.'+args.scan.split('.')[-1], "_fitted."+args.scan.split('.')[-1])
    parms = run_fitting(obj_path, lmk_path, out_name)

    # 把得到的flame再旋转平移回去
    logger.info("move the fitted flame back to original position")
    mesh = Mesh(filename=out_name)
    v = mesh.v
    # scale
    v /= parms["scale_factor"]
    # 平移
    v = transl(v, trans, 0)
    # 旋转
    v = x_rotate(v)
    write_simple_obj(v, mesh.f, out_name.replace('.obj', '_moveback.obj'))
    logger.info(f"check move-backed .ply at {out_name.replace('.obj', '_moveback.obj')}")

    parms["basic_trans"] = trans
    for k, v in parms.items():
        parms[k] = v.tolist()
    json_str = json.dumps(parms, indent=4)
    with open(out_name.replace('.obj', ".json"), 'w') as json_file:
        json_file.write(json_str)
    
    logger.info("==========================end============================")



if __name__ == '__main__':
    args = get_config()
    if args.fit_dir:
        run_fitting_dir(args)
    else:
        scan_path = args.scan
        scan_lmk_path = args.lmk
        # out_name = scan_path.replace('.'+args.scan.split('.')[-1], "_fitted."+args.scan.split('.')[-1])
        # run_fitting(scan_path, scan_lmk_path, out_name)


        # fit colmap result
        global logger
        logger = init_logging(save_path=Path(args.scan).parent)

        colmap_fitting(scan_path, scan_lmk_path)



'''
    如果需要fit colmap的输出
    只需要在本地标好关键点后，调用这里的 colmap_fitting() 即可
'''