#!/usr/bin/env python3

import os
import time
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from skimage.transform import radon
import yaml
import message_filters

from dre.msg import LoopCandidate
from dre.msg import LocalMapInfo


@dataclass
class MapEntry:
    index: int
    timestamp_us: int
    image_path: str
    sinofft: np.ndarray
    # Conjugate of the FFT of the sinofft along the first axis (computed once, used for the matching)
    sinofft_fft_conj: np.ndarray = None


class RaplaceNode(Node):
    def __init__(self) -> None:
        super().__init__("raplace_node")
    
        # Read the parameters from config/config_raplace.yaml
        config_file_path = "config/config_raplace.yaml"
        if not os.path.isfile(config_file_path):
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_file_path = os.path.join(base_path, "share/dre", config_file_path)
            self.output_dir = os.path.join(base_path, "share/dre", "raplace_local_maps")
        else:
            self.output_dir = "raplace_local_maps"
        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
        self.max_img_size = config["max_img_size"]
        self.min_time_diff = config["min_time_diff"]
        self.max_odom_drift = config["max_odom_drift"]
        self.pix_res = None
        
        self.down_shape = 0.6

        # Ensure the output directory exists and is empty
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        else:
            os.makedirs(self.output_dir, exist_ok=True)

        self.entries: List[MapEntry] = []
        self.theta = np.arange(0, 180)

        # Behavior branches on /operational_mode (published by mode_node, latched):
        # in "localization" mode, self.entries only ever holds place_candidate
        # library entries (see placeCandidateCallback), and live DRO scans are
        # only ever queried against them (queryAgainstLibrary), never inserted
        # into history or used for mapping-style loop closure. Every other mode
        # keeps the original insert-and-query-against-history behavior
        # (raplaceCallback), where self.entries only ever holds regular online
        # entries. mode_node fully restarts this node on every mode switch, so
        # self.entries is never a mix of the two — defaults to the original
        # behavior until a mode message actually arrives.
        self.mode_ = None
        mode_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_sub_ = self.create_subscription(String, "/operational_mode", self.modeCallback, mode_qos)

        # Create a synchronous subscription with larger queue and allow_headerless=False for strict sync
        self.subs = []
        self.subs.append(message_filters.Subscriber(self, Image, "dro_local_map_image"))
        self.subs.append(message_filters.Subscriber(self, LocalMapInfo, "dro_local_map_info"))
        self.ts = message_filters.TimeSynchronizer(self.subs, 2)
        self.ts.registerCallback(self.raplaceCallback)

        # Pre-built library entries (e.g. map chunks from initial_pose_selector's
        # init_mode="auto"): inserted into history so a live scan can be matched
        # against them, but never themselves used as a query against history.
        # KEEP_ALL so a fast burst of chunks isn't truncated while this callback's
        # Radon transform (the slow part) works through the backlog.
        place_candidate_qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_ALL)
        self.place_candidate_sub = self.create_subscription(
            Image, "place_candidate", self.placeCandidateCallback, place_candidate_qos
        )

        self.cumulated_dists = []
        self.times = []
        self.odome_poses = []

        self.candidate_pub = self.create_publisher(LoopCandidate, "raplace_loop_candidate", 10)

        self.get_logger().info(
            f"RaPlace online node started. Subscribed to 'dro_local_map_image' and 'dro_odometry'. Publishing candidates on 'raplace_loop_candidate'."
        )

    def modeCallback(self, msg: String):
        self.mode_ = msg.data
        self.get_logger().info(f"Operational mode set to '{self.mode_}'.")

    def timestampToFileName(self, timestamp_us: int) -> str:
        return os.path.join(self.output_dir, f"{timestamp_us}.png")


    @staticmethod
    def timestamp2us(msg: Image) -> int:
        return int(msg.header.stamp.sec) * 1_000_000 + int(msg.header.stamp.nanosec) // 1_000

    @staticmethod
    def fastDft(m_query: np.ndarray, m_item: np.ndarray) -> float:
        f_query = np.fft.fft(m_query, axis=0)
        f_item = np.fft.fft(m_item, axis=0)
        corrmap_2d = np.fft.ifft(f_query * np.conj(f_item), axis=0)
        corrmap = np.sum(corrmap_2d, axis=-1)
        maxval = np.max(corrmap)
        return float(np.real(maxval))

    # Same scores as fastDft(m_query, item) for a batch of items, given the conjugated FFTs of the items
    # (K, rows, cols): by linearity, the sum over the columns of the inverse FFTs is the inverse FFT of the
    # sum over the columns, so a single 1D inverse FFT per item is needed
    @staticmethod
    def fastDftBatch(m_query: np.ndarray, items_fft_conj: np.ndarray) -> np.ndarray:
        f_query = np.fft.fft(m_query, axis=0)
        corrmap = np.fft.ifft(np.einsum('rc,krc->kr', f_query, items_fft_conj), axis=1)
        return np.max(np.real(corrmap), axis=1)

    @staticmethod
    def sinofftEntry(index: int, timestamp_us: int, image_path: str, sinofft: np.ndarray) -> MapEntry:
        return MapEntry(index=index, timestamp_us=timestamp_us, image_path=image_path, sinofft=sinofft,
                        sinofft_fft_conj=np.conj(np.fft.fft(sinofft, axis=0)))

    # Best entry among the given ones for the query (the first one in case of equal scores), its score,
    # and the difference with the score of the query with itself
    def bestMatch(self, query_entry: MapEntry, candidates: List[MapEntry]):
        query_norm = (query_entry.sinofft - np.mean(query_entry.sinofft)) / (np.std(query_entry.sinofft) + 1e-8)
        scores = self.fastDftBatch(query_norm, np.stack([entry.sinofft_fft_conj for entry in candidates]))
        best = int(np.argmax(scores))
        self_score = float(self.fastDftBatch(query_norm, np.conj(np.fft.fft(query_norm, axis=0))[np.newaxis])[0])
        return candidates[best], float(scores[best]), abs(self_score - float(scores[best]))


    def computeSinofft(self, img_u8: np.ndarray) -> np.ndarray:
        if img_u8.shape[0] > self.max_img_size:
            scale_factor = self.max_img_size / img_u8.shape[0]
            new_width = int(img_u8.shape[1] * scale_factor)
            img_u8 = cv2.resize(img_u8, (new_width, self.max_img_size))
        elif img_u8.shape[1] > self.max_img_size:
            scale_factor = self.max_img_size / img_u8.shape[1]
            new_height = int(img_u8.shape[0] * scale_factor)
            img_u8 = cv2.resize(img_u8, (self.max_img_size, new_height))

        r = radon(img_u8, self.theta)
        max_r = float(np.max(r))
        if max_r > 1e-8:
            r = r / max_r

        r = r.astype(np.float64)
        r = cv2.resize(
            r,
            (int(self.down_shape * r.shape[1]), int(self.down_shape * r.shape[0]))
        )

        sinofft = np.abs(np.fft.fft(r, axis=0))
        return sinofft[: sinofft.shape[0] // 2, :]

    def saveLocalMap(self, img_u8: np.ndarray, timestamp_us: int) -> str:
        file_path = self.timestampToFileName(timestamp_us)
        cv2.imwrite(file_path, img_u8)
        return file_path

    def findBestCandidate(self, query_entry: MapEntry, odom_pose: np.ndarray):
        # Only compare with entries that are sufficiently far in time and space.
        # self.entries only ever holds regular online entries when this is called
        # (see the class docstring comment in __init__) — no library-entry bypass
        # needed here any more.
        time_mask = np.array(self.times) < (query_entry.timestamp_us - self.min_time_diff * 1e6)
        if not np.any(time_mask):
            return None
        dists = np.linalg.norm(np.array(self.odome_poses)[:,:2] - odom_pose[:2], axis=1)
        space_mask = dists < (self.max_odom_drift * (self.cumulated_dists[-1] - np.array(self.cumulated_dists)) + 50.0)
        valid_mask = np.logical_and(time_mask, space_mask)

        if not np.any(valid_mask):
            return None

        valid_ids = np.where(valid_mask)[0]
        return self.bestMatch(query_entry, [self.entries[idx] for idx in valid_ids])

    def findBestLibraryMatch(self, query_entry: MapEntry):
        # In localization mode, self.entries only ever holds place_candidate
        # library entries — no odometry, so no time/space gating applies or
        # makes sense; every entry is a valid candidate.
        if not self.entries:
            return None
        return self.bestMatch(query_entry, self.entries)



    def publishCandidate(self, query_entry: MapEntry, candidate_entry: MapEntry, score: float, min_dist: float, source_msg: Image):
        out = LoopCandidate()
        out.header = source_msg.header
        out.query_time = int(query_entry.timestamp_us)
        out.candidate_time = int(candidate_entry.timestamp_us)
        out.query_index = int(query_entry.index)
        out.candidate_index = int(candidate_entry.index)
        out.score = float(score)
        out.min_dist = float(min_dist)
        out.query_image_path = query_entry.image_path
        out.candidate_image_path = candidate_entry.image_path
        out.resolution = float(self.pix_res) if self.pix_res is not None else -1.0
        self.candidate_pub.publish(out)
        self.get_logger().info(
            f"Published candidate: query_idx={query_entry.index} candidate_idx={candidate_entry.index} "
            f"query_t={query_entry.timestamp_us} candidate_t={candidate_entry.timestamp_us} "
            f"score={score:.3f} min_dist={min_dist:.3f}"
        )

    def raplaceCallback(self, image_msg: Image, info_msg: LocalMapInfo):
        # Get the resolution of the local map from the first message
        if self.pix_res is None:
            self.pix_res = info_msg.resolution
            self.get_logger().info(f"Set pixel resolution to {self.pix_res} m/px based on the first received LocalMapInfo message.")

        image_np = np.frombuffer(image_msg.data, dtype=np.uint8).reshape((image_msg.height, image_msg.width))
        timestamp_us = self.timestamp2us(image_msg)

        if self.mode_ == "localization":
            self.queryAgainstLibrary(image_msg, image_np, timestamp_us)
            return

        # Compute the cumulated distance based on the odometry info
        xy = np.array([info_msg.x, info_msg.y])
        if len(self.cumulated_dists) == 0:
            self.cumulated_dists.append(0.0)
        else:
            dist = np.linalg.norm(xy - self.odome_poses[-1][:2])
            self.cumulated_dists.append(dist + (self.cumulated_dists[-1]))
        self.odome_poses.append(np.array([info_msg.x, info_msg.y, info_msg.theta]))

        odom_pose = np.array([info_msg.x, info_msg.y, info_msg.theta])

        # Create a new MapEntry (including computing the sinofft)
        map_entry = self.sinofftEntry(
            index=len(self.entries),
            timestamp_us=timestamp_us,
            image_path=self.timestampToFileName(timestamp_us),
            sinofft=self.computeSinofft(image_np),
        )
        self.entries.append(map_entry)
        self.times.append(timestamp_us)

        # Save the local map image to disk
        self.saveLocalMap(image_np, timestamp_us)

        # Find the best candidate for loop closure and publish it if it exists
        best_match = self.findBestCandidate(map_entry, odom_pose)
        if best_match is not None:
            best_entry, best_score, min_dist = best_match
            self.publishCandidate(map_entry, best_entry, best_score, min_dist, image_msg)

    def queryAgainstLibrary(self, image_msg: Image, image_np: np.ndarray, timestamp_us: int):
        # Transient query: never added to self.entries, so it never pollutes the
        # library and never gets matched against by a later query. Still saved to
        # disk under a real path — registration_node loads images by path, not
        # from the message itself.
        image_path = self.saveLocalMap(image_np, timestamp_us)
        query_entry = self.sinofftEntry(
            index=-1,
            timestamp_us=timestamp_us,
            image_path=image_path,
            sinofft=self.computeSinofft(image_np),
        )

        best_match = self.findBestLibraryMatch(query_entry)
        if best_match is not None:
            best_entry, best_score, min_dist = best_match
            self.publishCandidate(query_entry, best_entry, best_score, min_dist, image_msg)
        else:
            self.get_logger().warn(
                "Live query received but the place_candidate library is empty "
                "(chunks not published/received yet?) — skipping."
            )

    def placeCandidateCallback(self, image_msg: Image):
        image_np = np.frombuffer(image_msg.data, dtype=np.uint8).reshape((image_msg.height, image_msg.width))
        timestamp_us = self.timestamp2us(image_msg)

        map_entry = self.sinofftEntry(
            index=len(self.entries),
            timestamp_us=timestamp_us,
            image_path=self.timestampToFileName(timestamp_us),
            sinofft=self.computeSinofft(image_np),
        )
        self.entries.append(map_entry)

        self.saveLocalMap(image_np, timestamp_us)

        self.get_logger().info(f"Inserted place_candidate into history (library size={len(self.entries)}).")





def main(args=None):
    rclpy.init(args=args)
    node = RaplaceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
