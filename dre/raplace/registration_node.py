#!/usr/bin/env python3

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
import yaml
import torch
import time
from scipy.spatial.transform import Rotation as R

from dre.msg import LoopCandidate


@dataclass
class RegistrationResult:
    valid: bool
    pose: np.ndarray
    scale: float
    num_matches: int
    reason: str
    viz_image: Optional[np.ndarray]

def poseToxytheta(pose):
    x = pose[0, 3]
    y = pose[1, 3]
    theta = np.arctan2(pose[1, 0], pose[0, 0])
    return np.array([x, y, theta])

def xythetaToPose(xytheta):
    c_rot = np.cos(xytheta[2])
    s_rot = np.sin(xytheta[2])
    pose = np.eye(4)
    pose[0, 0] = c_rot
    pose[0, 1] = -s_rot
    pose[1, 0] = s_rot
    pose[1, 1] = c_rot
    pose[0, 3] = xytheta[0]
    pose[1, 3] = xytheta[1]
    return pose

def affineToPoseAndScale(affine_matrix, pix_res, img_shape):
    # Transform to convert the opencv frame to the local map frame
    T_cv_local_map = np.array([[0, 1, 0, pix_res*img_shape[1]/2],
                               [-1, 0, 0, pix_res*img_shape[0]/2],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
    T_local_map_cv = np.linalg.inv(T_cv_local_map)

    # Get the scale from the affine matrix
    scale = np.linalg.norm(affine_matrix[0, :2])
    # Get the rotation from the affine matrix
    rotation = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])
    pose = np.eye(4)
    pose[0, 0] = np.cos(rotation)
    pose[0, 1] = -np.sin(rotation)
    pose[1, 0] = np.sin(rotation)
    pose[1, 1] = np.cos(rotation)
    pose[0, 3] = affine_matrix[0, 2] * pix_res
    pose[1, 3] = affine_matrix[1, 2] * pix_res

    # Convert the pose to the local map frame
    pose = T_local_map_cv @ pose @ T_cv_local_map
    return pose, scale

class LocalMapRegistrator:
    # Direct registration of the source image in the target image: maximisation of the correlation
    # sum_p source(p) * target(T(p)) w.r.t. the 2D pose (x, y, theta) of the transformation T.
    # Only the non-zero pixels of the source contribute to the cost and its gradient, so the
    # computations are done on these pixels only (sparse).
    def __init__(self, source, target, res, xytheta_init=np.array([0, 0, 0]), use_gpu_if_available=True):

        # Check the input shapes match and that the nb of collumn and rows are odd
        if source.shape[0] != target.shape[0] or source.shape[1] != target.shape[1] or source.shape[0] % 2 == 0 or source.shape[1] % 2 == 0:
            raise ValueError("Source and target images must have the same shape and odd dimensions")

        if use_gpu_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(1)

        self.optimisation_first_step = 0.1

        with torch.no_grad():
            self.source = torch.tensor(source, device=self.device).float()
            self.target = torch.tensor(target, device=self.device).float()
            self.res = float(res)
            self.xytheta_init = torch.tensor(xytheta_init, device=self.device).float()
            self.half_height = self.source.shape[0] // 2
            self.half_width = self.source.shape[1] // 2

            # Cartesian coordinates and values of the non-zero pixels of the source
            # (row r is at x = -(r - H//2)*res, column c is at y = (c - W//2)*res)
            rows, cols = torch.nonzero(self.source, as_tuple=True)
            self.source_values = self.source[rows, cols]
            self.source_x = -(rows - self.half_height).float() * self.res
            self.source_y = (cols - self.half_width).float() * self.res


    # Rotated source coordinates and their (row, column) coordinates in the target image for the
    # pose(s) xytheta (shape (3,) or (K, 3), giving (N,) or (K, N) coordinates)
    def transformSparse_(self, xytheta):
        x, y, theta = xytheta[..., 0:1], xytheta[..., 1:2], xytheta[..., 2:3]
        c_rot = torch.cos(theta)
        s_rot = torch.sin(theta)
        x_rot = c_rot * self.source_x - s_rot * self.source_y
        y_rot = s_rot * self.source_x + c_rot * self.source_y
        row = (x_rot + x) / (-self.res) + self.half_height
        col = (y_rot + y) / self.res + self.half_width
        return x_rot, y_rot, row, col


    # Bilinear interpolation of the image im at (row, col), with the coordinates clamped to the image,
    # and the derivatives of the interpolated values w.r.t. the row and column
    def bilinearInterpolationSparse_(self, im, row, col, with_jac=False):
        max_row = im.shape[0] - 1
        max_col = im.shape[1] - 1
        row0 = torch.floor(row).long()
        col0 = torch.floor(col).long()
        row1 = torch.clamp(row0 + 1, 0, max_row)
        col1 = torch.clamp(col0 + 1, 0, max_col)
        row0 = torch.clamp(row0, 0, max_row)
        col0 = torch.clamp(col0, 0, max_col)
        row = torch.clamp(row, 0, max_row)
        col = torch.clamp(col, 0, max_col)

        Ia = im[row0, col0]
        Ib = im[row1, col0]
        Ic = im[row0, col1]
        Id = im[row1, col1]

        one_minus_col = col1.float() - col
        local_col = col - col0.float()
        one_minus_row = row1.float() - row
        local_row = row - row0.float()
        interp = (one_minus_row * one_minus_col) * Ia + (local_row * one_minus_col) * Ib + (one_minus_row * local_col) * Ic + (local_row * local_col) * Id
        if not with_jac:
            return interp
        d_interp_d_row = (Ib - Ia) * one_minus_col + (Id - Ic) * local_col
        d_interp_d_col = (Ic - Ia) * one_minus_row + (Id - Ib) * local_row
        return interp, d_interp_d_row, d_interp_d_col


    # Cost (sum of the residuals source * interpolated target) and its gradient w.r.t. the pose
    def costAndGradient(self, xytheta):
        with torch.no_grad():
            x_rot, y_rot, row, col = self.transformSparse_(xytheta)
            interp, d_row, d_col = self.bilinearInterpolationSparse_(self.target, row, col, with_jac=True)
            cost = torch.sum(interp * self.source_values)
            # d row / d (x, y, theta) = -1/res * (1, 0, -y_rot), d col / d (x, y, theta) = 1/res * (0, 1, x_rot)
            weighted_d_row = (d_row * self.source_values) * (-1.0 / self.res)
            weighted_d_col = (d_col * self.source_values) * (1.0 / self.res)
            grad = torch.stack((torch.sum(weighted_d_row), torch.sum(weighted_d_col),
                                torch.sum(weighted_d_col * x_rot - weighted_d_row * y_rot)))
            return cost, grad


    # Cost for a batch of poses (K, 3)
    def batchCost(self, xythetas):
        with torch.no_grad():
            _, _, row, col = self.transformSparse_(xythetas)
            return torch.sum(self.bilinearInterpolationSparse_(self.target, row, col) * self.source_values, dim=-1)


    # Gradient ascent with a normalised step, halved (and going back to the last increasing state)
    # when the cost decreases. The branches are evaluated on the device so that there is a single
    # device-host synchronisation per iteration (for the stopping criteria).
    def register(self, nb_iter=20, cost_tol=1e-6, step_tol=1e-6):
        with torch.no_grad():
            state = self.xytheta_init.clone().to(self.device).float()
            prev_cost = torch.tensor(np.inf, device=self.device)
            step_quantum = torch.tensor(self.optimisation_first_step, device=self.device)
            last_increasing_state = state.clone()
            last_increasing_grad = torch.zeros_like(state)
            for i in range(nb_iter):
                cost, grad = self.costAndGradient(state)

                if i == 0:
                    last_increasing_grad = grad.clone()
                else:
                    decreased = cost < prev_cost
                    state = torch.where(decreased, last_increasing_state, state)
                    grad = torch.where(decreased, last_increasing_grad, grad)
                    step_quantum = torch.where(decreased, step_quantum / 2, step_quantum)
                    last_increasing_state = state.clone()
                    last_increasing_grad = grad.clone()

                grad_norm = torch.linalg.norm(grad)
                stop_before_step = (step_quantum < 1e-5) | (grad_norm < 1e-9)
                step = (grad / grad_norm) * step_quantum
                step_norm = torch.linalg.norm(step)
                stop_after_step = (step_norm < step_tol) | (torch.abs((cost - prev_cost) / cost) < cost_tol)

                stop_before_step, stop_after_step = torch.stack((stop_before_step, stop_after_step)).tolist()
                if stop_before_step:
                    break
                state = state + step
                if stop_after_step:
                    break
                prev_cost = cost

            state_np = state.detach().cpu().numpy()

            self.xytheta_init = state.clone()

            return state_np


    # Dense transformation of the source (for the visualisation only)
    def transformSource_(self, xytheta):
        with torch.no_grad():
            # If xytheta is a numpy array, convert it to a torch tensor
            if isinstance(xytheta, np.ndarray):
                xytheta = torch.tensor(xytheta, device=self.device).float()

            rows = torch.arange(self.source.shape[0], device=self.device).float().unsqueeze(1)
            cols = torch.arange(self.source.shape[1], device=self.device).float().unsqueeze(0)
            x = -(rows - self.half_height) * self.res
            y = (cols - self.half_width) * self.res

            # Inverse transformation: R^T (p - t)
            c_rot = torch.cos(xytheta[2])
            s_rot = torch.sin(xytheta[2])
            x_t = c_rot * (x - xytheta[0]) + s_rot * (y - xytheta[1])
            y_t = -s_rot * (x - xytheta[0]) + c_rot * (y - xytheta[1])
            row = x_t / (-self.res) + self.half_height
            col = y_t / self.res + self.half_width

            source_interp = self.bilinearInterpolationSparse_(self.source, row.flatten(), col.flatten()).reshape(self.source.shape)

            # Residuals
            residuals = source_interp * self.source

            return source_interp, residuals


    def displayOverlay(self, show=True):
        # Display the overlay of the source and target images
        source_interp, _ = self.transformSource_(self.xytheta_init)

        target_np = self.target.cpu().numpy()
        source_interp_np = source_interp.cpu().numpy()
        
        def normalizeToUint8(arr):
            a, b = arr.min(), arr.max()
            if b > a:
                arr = (arr - a) / (b - a)
            return (arr * 255).astype(np.uint8)

        target_uint8 = normalizeToUint8(target_np)
        source_uint8 = normalizeToUint8(source_interp_np)

        # Grayscale target → BGR
        target_bgr = cv2.cvtColor(target_uint8, cv2.COLOR_GRAY2BGR)

        # 'hot' colormap on source_interp
        source_hot = cv2.applyColorMap(source_uint8, cv2.COLORMAP_HOT)

        # alpha=0.5 blend: result = target * 0.5 + source_hot * 0.5
        overlay = cv2.addWeighted(target_bgr, 0.5, source_hot, 0.5, 0)

        if show:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10,10))
            # Swap the color channels from BGR to RGB for correct display in matplotlib
            overlay_dis = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            plt.imshow(overlay_dis)
            plt.title('Overlay of target (grayscale) and source (hot colormap)')
            plt.axis('off')
            plt.show()

        return overlay

    def getRegistrationScore(self):
        # Compute the registration
        with torch.no_grad():
            cost, _ = self.costAndGradient(self.xytheta_init)
            return cost / torch.sum(self.target**2)

    # Exhaustive search on a grid around the initial pose (all the poses evaluated in a single batch,
    # the first best pose in the x, y, theta order is kept)
    def gridSearchInitialization(self, search_ranges, nb_steps):
        with torch.no_grad():
            xs = torch.linspace(search_ranges[0][0], search_ranges[0][1], nb_steps, device=self.device) + self.xytheta_init[0]
            ys = torch.linspace(search_ranges[1][0], search_ranges[1][1], nb_steps, device=self.device) + self.xytheta_init[1]
            thetas = torch.linspace(search_ranges[2][0], search_ranges[2][1], nb_steps, device=self.device) + self.xytheta_init[2]
            candidates = torch.stack(torch.meshgrid(xs, ys, thetas, indexing='ij'), dim=-1).reshape((-1, 3))
            costs = self.batchCost(candidates)
            self.xytheta_init = candidates[torch.argmax(costs)].clone()

class RegistrationNode(Node):
    def __init__(self) -> None:
        super().__init__("registration_node")

        config_file_path = "config/config_registration.yaml"
        self.package_share = ""
        if not os.path.isfile(config_file_path):
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.package_share = os.path.join(base_path, "share", "dre")
            config_file_path = os.path.join(self.package_share, config_file_path)
        with open(config_file_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            

        self.lowe_ratio = float(cfg["lowe_ratio"])
        self.ransac_thr = float(cfg["ransac_thr"])
        self.max_img_size = int(cfg["max_img_size"])

        if "use_gpu_if_available" in cfg:
            self.use_gpu_if_available = bool(cfg["use_gpu_if_available"])
        else:
            self.use_gpu_if_available = True

        self.max_scale_error = 0.05
        self.sift_extractor = cv2.SIFT_create(
            nfeatures=0,
            contrastThreshold=0.02,
            edgeThreshold=20,
            sigma=2.5,
        )
        self.sift_matcher = cv2.BFMatcher()
        self.bridge = CvBridge()

        self.candidate_sub = self.create_subscription(
            LoopCandidate,
            "raplace_loop_candidate",
            self.candidateCallback,
            20,
        )
        self.pose_pub = self.create_publisher(TransformStamped, "registration_relative_pose", 20)
        self.viz_pub = self.create_publisher(Image, "registration_debug_image", 20)

        self.get_logger().info(
            "Registration node started. Subscribed to 'raplace_loop_candidate', publishing 'registration_relative_pose' and 'registration_debug_image'."
        )

        self.counter = 0

    def resolvePath(self, path: str) -> Optional[str]:
        if os.path.isabs(path) and os.path.isfile(path):
            return path
        if os.path.isfile(path):
            return os.path.abspath(path)
        if self.package_share:
            candidate = os.path.join(self.package_share, path)
            if os.path.isfile(candidate):
                return candidate
        return None

    def candidateCallback(self, msg: LoopCandidate) -> None:
        query_path = self.resolvePath(msg.query_image_path)
        candidate_path = self.resolvePath(msg.candidate_image_path)

        if query_path is None or candidate_path is None:
            self.get_logger().warn(
                f"Skipping candidate q={msg.query_index} c={msg.candidate_index}: image path not found "
                f"(query='{msg.query_image_path}', candidate='{msg.candidate_image_path}')."
            )
            return

        query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
        candidate_img = cv2.imread(candidate_path, cv2.IMREAD_GRAYSCALE)
        if query_img is None or candidate_img is None:
            self.get_logger().warn(
                f"Skipping candidate q={msg.query_index} c={msg.candidate_index}: failed to read image files."
            )
            return

        result = self.estimateRelativePose(query_img, candidate_img, float(msg.resolution))

        if result.valid:
            t1 = time.time()
            result = self.refineRegistration(candidate_img, query_img, float(msg.resolution), result)
            t2 = time.time()
            self.get_logger().info(
                f"Refinement for candidate q={msg.query_index} c={msg.candidate_index} took {(t2-t1) * 1000:.1f} ms. Final reason: {result.reason}."
            )

        if result.valid:
            self.get_logger().info(
                f"Candidate q={msg.query_index} c={msg.candidate_index} registered successfully: "
                f"pose={poseToxytheta(result.pose)}, scale={result.scale:.3f}, matches={result.num_matches}, reason={result.reason}."
            )
        else:
            # Most candidates fail this way (a routine "not a match", not a
            # problem) — kept at INFO for now (devel visibility), same line
            # publishResult() uses for the "accepted" case.
            self.get_logger().info(
                f"Candidate q={msg.query_index} c={msg.candidate_index} registration failed: "
                f"scale={result.scale:.3f}, matches={result.num_matches}, reason={result.reason}."
            )
            return

        self.publishResult(msg, result)




    def estimateRelativePose(
        self,
        query_img: np.ndarray,
        candidate_img: np.ndarray,
        resolution_m_per_px: float,
    ) -> RegistrationResult:
        original_shape = query_img.shape
        ratio = 1.0
        img2 = query_img.copy()
        img1 = candidate_img.copy()

        if img2.shape[0] > self.max_img_size:
            ratio = img2.shape[0] / float(self.max_img_size)
            img2 = cv2.resize(img2, (self.max_img_size, self.max_img_size))
            img1 = cv2.resize(img1, (self.max_img_size, self.max_img_size))

        kp_2, des_2 = self.sift_extractor.detectAndCompute(img2, None)
        kp_1, des_1 = self.sift_extractor.detectAndCompute(img1, None)
        if des_2 is None or des_1 is None:
            return RegistrationResult(False, None, 1.0, 0, "missing_descriptors", None)

        if ratio != 1.0:
            for kp in kp_2:
                kp.pt = (kp.pt[0] * ratio, kp.pt[1] * ratio)
            for kp in kp_1:
                kp.pt = (kp.pt[0] * ratio, kp.pt[1] * ratio)

        matches = self.sift_matcher.knnMatch(des_1, des_2, k=2)

        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if (kp_1[m.queryIdx].octave & 255) != (kp_2[n.trainIdx].octave & 255):
                continue
            if m.distance < self.lowe_ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            return RegistrationResult(False, None, 1.0, 0, "insufficient_matches", None)

        dst_pts = np.float32([kp_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thr,
        )
        if M is None:
            return RegistrationResult(False, None, 1.0, len(good_matches), "ransac_failed", None)
        
        if inliers is None or np.sum(inliers) < 3:
            return RegistrationResult(False, None, 1.0, len(good_matches), "insufficient_inliers", None)


        pose, scale = affineToPoseAndScale(M, resolution_m_per_px, original_shape)


        if abs(scale - 1.0) > self.max_scale_error:
            return RegistrationResult(False, None, scale, len(good_matches), "scale_error", None)


        M_viz = M.copy()
        M_viz[:, -1] = M_viz[:, -1] / ratio
        img2_warped = cv2.warpAffine(img2, M_viz, (img1.shape[1], img1.shape[0]))

        if inliers is not None:
            inlier_ratio = float(np.mean(inliers))
            reason = f"ok_inlier_ratio_{inlier_ratio:.2f}"
        else:
            reason = "ok"

        self.counter += 1

        return RegistrationResult(
            True,
            pose,
            scale,
            len(good_matches),
            reason,
            None
        )
    
    def refineRegistration(self, img_i, img_j, res, reg_result):
            if img_i is None or img_j is None:
                self.get_logger().warn("Skipping registration due to missing images.")
                return
            img_i = cv2.GaussianBlur(img_i, (5, 5), 0)
            img_j = cv2.GaussianBlur(img_j, (5, 5), 0)

            # Resize the images
            img_i_small = cv2.resize(img_i, (img_i.shape[1]//4 + 1, img_i.shape[0]//4 + 1))
            img_j_small = cv2.resize(img_j, (img_j.shape[1]//4 + 1, img_j.shape[0]//4 + 1))
            res_small = res * 4
            # Add Gaussian blur to the images
            img_i_small = cv2.GaussianBlur(img_i_small, (5, 5), 0)
            img_j_small = cv2.GaussianBlur(img_j_small, (5, 5), 0)

            # Perform fine registration using "gp_doppler"
            local_map_registrator = LocalMapRegistrator(img_j_small, img_i_small, res_small, poseToxytheta(reg_result.pose), use_gpu_if_available=self.use_gpu_if_available)
            local_map_registrator.gridSearchInitialization([[-2,2],[-2,2], [np.radians(-1.0), np.radians(1.0)]], nb_steps=3)

            state = local_map_registrator.register(nb_iter=20, step_tol=1e-4)

            # Resize the images
            img_i_small = cv2.resize(img_i, (img_i.shape[1]//2 + 1, img_i.shape[0]//2 + 1))
            img_j_small = cv2.resize(img_j, (img_j.shape[1]//2 + 1, img_j.shape[0]//2 + 1))
            res_small = res * 2
            # Add Gaussian blur to the images
            img_i_small = cv2.GaussianBlur(img_i_small, (5, 5), 0)
            img_j_small = cv2.GaussianBlur(img_j_small, (5, 5), 0)
            local_map_registrator = LocalMapRegistrator(img_j_small, img_i_small, res_small, state, use_gpu_if_available=self.use_gpu_if_available)
            state = local_map_registrator.register(nb_iter=40, step_tol=1e-4)
            x, y, theta = state

            reg_score = local_map_registrator.getRegistrationScore()
            if reg_score > 0.5:
                new_result = RegistrationResult(
                    True,
                    xythetaToPose(state),
                    reg_result.scale,
                    reg_result.num_matches,
                    f"refined_{reg_result.reason}",
                    # Only needed for the debug image (published only when subscribed)
                    local_map_registrator.displayOverlay(show=False) if self.viz_pub.get_subscription_count() > 0 else None
                )
            else:
                new_result = RegistrationResult(
                    False,
                    reg_result.pose,
                    reg_result.scale,
                    reg_result.num_matches,
                    f"refinement_failed_score_{reg_score:.2f}",
                    None
                )
            return new_result


    def publishResult(self, source_msg: LoopCandidate, result: RegistrationResult) -> None:
        if result.valid:
            x_m = result.pose[0, 3]
            y_m = result.pose[1, 3]
            theta_rad = np.arctan2(result.pose[1, 0], result.pose[0, 0])
            pose_msg = TransformStamped()
            pose_msg.header = source_msg.header
            pose_msg.header.frame_id = str(source_msg.candidate_time)
            pose_msg.child_frame_id = str(source_msg.query_time)
            pose_msg.transform.translation.x = x_m
            pose_msg.transform.translation.y = y_m
            pose_msg.transform.translation.z = 0.0
            quat = R.from_matrix(result.pose[:3, :3]).as_quat()
            pose_msg.transform.rotation.x = quat[0]
            pose_msg.transform.rotation.y = quat[1]
            pose_msg.transform.rotation.z = quat[2]
            pose_msg.transform.rotation.w = quat[3]
            self.pose_pub.publish(pose_msg)


            if self.viz_pub.get_subscription_count() > 0 and result.viz_image is not None:
                viz = result.viz_image.copy()
                status = "ACCEPTED" if result.valid else "REJECTED"
                text = (
                    f"{status} | q={source_msg.query_index} c={source_msg.candidate_index} "
                    f"| m={result.num_matches} | s={result.scale:.3f} | {result.reason}"
                )
                cv2.putText(
                    viz,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if result.valid else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                image_msg = self.bridge.cv2_to_imgmsg(viz, encoding="bgr8")
                image_msg.header = source_msg.header
                self.viz_pub.publish(image_msg)

            self.get_logger().info(
                f"Registration accepted q={source_msg.query_index} c={source_msg.candidate_index}: "
                f"x={x_m:.2f} m, y={y_m:.2f} m, theta={theta_rad:.3f} rad, "
                f"matches={result.num_matches}, scale={result.scale:.3f}"
            )
        else:
            self.get_logger().debug(
                f"Registration rejected q={source_msg.query_index} c={source_msg.candidate_index}: "
                f"reason={result.reason}, matches={result.num_matches}, scale={result.scale:.3f}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RegistrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
