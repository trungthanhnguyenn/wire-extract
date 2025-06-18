import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from electric_line.seg_de import Segmentation
from collections import defaultdict
import math

class LengthCalculator:
    def __init__(self, seg: Segmentation):
        self.seg = seg
        self.valid_connections = []
        self.cached_mask = None

    def _get_points_on_line(self, p1, p2, dense=False):
        """Get points on line between p1 and p2"""
        x1, y1 = p1
        x2, y2 = p2
        distance = np.linalg.norm(np.array(p1) - np.array(p2))
        
        if dense:
            num_points = max(20, int(distance))  # Denser sampling for overlap detection
        else:
            num_points = max(10, int(distance * 1.5))
            
        points = []
        for t in np.linspace(0, 1, num_points):
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            points.append((x, y))
        return points

    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line segment"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line_start to line_end
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return np.linalg.norm(np.array(point) - np.array(line_start))
        
        # Parameter t for closest point on line
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
        
        # Closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return np.linalg.norm(np.array(point) - np.array([closest_x, closest_y]))

    def _is_point_on_line_segment(self, point, line_start, line_end, tolerance=5):
        """Check if point lies on line segment within tolerance"""
        distance = self._point_to_line_distance(point, line_start, line_end)
        return distance <= tolerance

    def calculate_circuit_length(self, cluster_tolerance=8, line_tolerance=2, overlap_threshold=0.7, use_skeleton=False):
        """
        Calculate circuit length with improved overlap detection
        
        Args:
            cluster_tolerance: Distance threshold for clustering nearby points
            line_tolerance: Tolerance for skeleton matching
            overlap_threshold: Threshold for determining line overlap (0.0-1.0)
            use_skeleton: Whether to apply thinning before finding paths
        """
        # Load or generate mask
        mask_path = os.path.join(self.seg.output_dir, f"masks_{os.path.basename(self.seg.image_path)}")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = self.seg.visual()
        self.cached_mask = mask

        # Apply skeleton if needed
        if use_skeleton:
            try:
                # import cv2.ximgproc
                self.skeleton = cv2.ximgproc.thinning(mask)
            except ImportError:
                raise ImportError("Please install opencv-contrib-python to use skeleton thinning.")
        else:
            self.skeleton = mask

        # Load center points
        _, txt_path = self.seg.detect_circle()
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"File center point does not exist: {txt_path}")

        center_points = self._load_center_points(txt_path)
        if len(center_points) < 2:
            return 0, [], []

        # Cluster nearby points
        points_array = self._cluster_points(center_points, cluster_tolerance)

        # Find all valid connections
        all_connections = self._find_valid_connections(points_array, self.skeleton, line_tolerance)

        # Remove overlapping connections with improved algorithm
        filtered_connections = self._remove_overlapping_connections_improved(all_connections, overlap_threshold)

        # Convert to final format
        optimized_connections = self._remove_duplicate_connections(filtered_connections)

        # Store for visualization
        self.valid_connections = filtered_connections

        total_length = sum(np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in optimized_connections)
        return total_length, list(optimized_connections), self.valid_connections

    def _load_center_points(self, txt_path):
        """Load center points from text file"""
        points = []
        with open(txt_path, 'r') as f:
            for line in f:
                if "Center = " in line:
                    coords_str = line.strip().split("Center = ")[1].strip("()")
                    x, y = map(int, coords_str.split(","))
                    points.append((x, y))
        return points

    def _cluster_points(self, points, tolerance):
        """Cluster nearby points to reduce noise"""
        if tolerance <= 0:
            return np.array(points)
            
        points_array = np.array(points)
        clustering = DBSCAN(eps=tolerance, min_samples=1).fit(points_array)
        
        clustered = []
        for label in np.unique(clustering.labels_):
            cluster = points_array[clustering.labels_ == label]
            center = tuple(cluster.mean(axis=0).astype(int))
            clustered.append(center)
            
        return np.array(clustered)

    def _find_valid_connections(self, points_array, skeleton, tolerance):
        """Find all valid connections between points"""
        valid = []
        for i in range(len(points_array)):
            for j in range(i + 1, len(points_array)):
                p1, p2 = points_array[i], points_array[j]
                if self._check_line_on_skeleton_fast(p1, p2, self.skeleton, tolerance):
                    dist = np.linalg.norm(np.array(p1) - np.array(p2))
                    valid.append((tuple(p1), tuple(p2), dist))
        return valid

    def _remove_overlapping_connections_improved(self, connections, overlap_threshold=0.7):
        """
        Improved algorithm to remove overlapping connections
        """
        if not connections:
            return []
            
        # Sort by length (shorter lines first for better accuracy)
        connections_sorted = sorted(connections, key=lambda x: x[2])
        
        valid_connections = []
        
        for current_conn in connections_sorted:
            p1, p2, dist = current_conn
            current_points = set(self._get_points_on_line(p1, p2, dense=True))
            
            is_overlapping = False
            
            # Check against already accepted connections
            for existing_conn in valid_connections:
                ep1, ep2, _ = existing_conn
                existing_points = set(self._get_points_on_line(ep1, ep2, dense=True))
                
                # Calculate overlap
                intersection = current_points & existing_points
                union = current_points | existing_points
                
                if len(union) > 0:
                    overlap_ratio = len(intersection) / len(union)
                    
                    # Additional check: if current line's endpoints lie on existing line
                    if (self._is_point_on_line_segment(p1, ep1, ep2, tolerance=8) and 
                        self._is_point_on_line_segment(p2, ep1, ep2, tolerance=8)):
                        is_overlapping = True
                        break
                    
                    # Check if existing line's endpoints lie on current line
                    if (self._is_point_on_line_segment(ep1, p1, p2, tolerance=8) and 
                        self._is_point_on_line_segment(ep2, p1, p2, tolerance=8)):
                        is_overlapping = True
                        break
                        
                    # Standard overlap check
                    if overlap_ratio > overlap_threshold:
                        is_overlapping = True
                        break
            
            if not is_overlapping:
                valid_connections.append(current_conn)
        
        return valid_connections

    def _remove_duplicate_connections(self, connections):
        """Remove duplicate connections"""
        seen = set()
        for p1, p2, _ in connections:
            key = tuple(sorted((tuple(p1), tuple(p2))))
            seen.add(key)
        return seen

    def _check_line_on_skeleton_fast(self, p1, p2, skeleton, tolerance):
        """Check if line follows skeleton pattern"""
        x1, y1 = p1
        x2, y2 = p2
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        samples = max(8, min(100, int(dist / 1.5)))

        valid = 0
        total_samples = 0
        
        for i in range(samples + 1):
            t = i / samples
            x, y = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
            
            # Skip if point is outside image bounds
            if self.skeleton is None:
                raise ValueError("Skeleton does not exist")
            if 0 <= x < self.skeleton.shape[1] and 0 <= y < self.skeleton.shape[0]:
                total_samples += 1
                if self._check_point_near_skeleton(x, y, self.skeleton, tolerance):
                    valid += 1
        
        if total_samples == 0:
            return False
            
        return (valid / total_samples) >= 0.5

    def _check_point_near_skeleton(self, x, y, skeleton, tolerance):
        """Check if point is near skeleton within tolerance"""
        h, w = self.skeleton.shape
        for dx in range(-tolerance, tolerance + 1):
            for dy in range(-tolerance, tolerance + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and self.skeleton[ny, nx] > 0:
                    return True
        return False

    def visualize_optimized_result(self, connections, save_path=None, show_all_points=False, highlight_conns=None):
        """
        Visualize the result with improved point display
        
        Args:
            connections: List of valid connections
            save_path: Path to save result image
            show_all_points: If True, show all detected points; if False, only show endpoints
        """
        result_img = self.seg.orig.copy()
        if self.cached_mask is None:
            self.cached_mask = self.seg.visual()

        # Create overlay with mask
        mask_colored = cv2.cvtColor(self.cached_mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(result_img, 0.7, mask_colored, 0.3, 0)
        
        # # Draw connections
        # for p1, p2 in connections:
        #     cv2.line(overlay, p1, p2, (255, 0, 0), 2)

        for p1, p2 in connections:
            color = (255, 0, 0)  # Blue by default
            if highlight_conns and tuple(sorted((p1, p2))) in highlight_conns:
                color = (0, 165, 255)  # Orange in BGR
            cv2.line(overlay, p1, p2, color, 2)

        # Show endpoints
        endpoint_set = set()
        for p1, p2 in connections:
            endpoint_set.add(p1)
            endpoint_set.add(p2)

        for point in endpoint_set:
            cv2.circle(overlay, point, 5, (0, 255, 0), -1)  # Green
    
        # Determine which points to show
        if show_all_points:
            # Show all detected center points
            _, txt_path = self.seg.detect_circle()
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        if 'Center = ' in line:
                            coords_str = line.strip().split("Center = ")[1].strip("()")
                            x, y = map(int, coords_str.split(", "))
                            cv2.circle(overlay, (x, y), 4, (0, 255, 255), -1)  # Yellow for all points

        if save_path:
            cv2.imwrite(save_path, overlay)
            
        return overlay

    def get_connection_statistics(self):
        """Get statistics about the connections"""
        if not self.valid_connections:
            return {}
            
        lengths = [conn[2] for conn in self.valid_connections]
        endpoint_count = len(set([p for conn in self.valid_connections for p in conn[:2]]))
        
        return {
            'total_connections': len(self.valid_connections),
            'total_length': sum(lengths),
            'average_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'unique_endpoints': endpoint_count
        }
        
    
    def get_initial_direction_mask(self, p, mask, radius=20):
        x0, y0 = p
        h, w = mask.shape
        directions = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x0 + dx, y0 + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] > 0:
                    directions.append((dx, dy))

        if directions:
            mean_dx = np.mean([d[0] for d in directions])
            mean_dy = np.mean([d[1] for d in directions])
            norm = np.linalg.norm([mean_dx, mean_dy])
            if norm == 0:
                return None
            return (mean_dx / norm, mean_dy / norm)
        return None


    def track_mask(self, p, direction, mask, max_length=2000, step=10, turn_threshold=15):
        path = [p]
        prev_dir = direction
        px, py = p
        h, w = mask.shape

        if direction is None:
            return p

        for _ in range(max_length):
            # If out of range, slowly move step to 1
            nx = int(round(px + direction[0] * step))
            ny = int(round(py + direction[1] * step))

            # Down step if near to end point
            dist_to_edge = min(px, py, w - px - 1, h - py - 1)
            if dist_to_edge < step:
                nx = int(round(px + direction[0]))
                ny = int(round(py + direction[1]))

            if not (0 <= nx < w and 0 <= ny < h):
                break
            if mask[ny, nx] == 0:
                break

            path.append((nx, ny))

            # Calculate new direction
            new_dir = self.get_initial_direction_mask((nx, ny), mask, radius=10)
            if new_dir:
                angle_diff = self.angle_between_vectors(prev_dir, new_dir)
                if angle_diff > turn_threshold:
                    break
                prev_dir = new_dir
                direction = new_dir
            else:
                break

            px, py = nx, ny

        return path[-1]
    
    def angle_between_vectors(self, v1, v2):
        unit_v1 = np.array(v1) / np.linalg.norm(v1)
        unit_v2 = np.array(v2) / np.linalg.norm(v2)
        dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def has_missing_direction(self, vectors, threshold=45):
        """Checking if direct vector has angel > 45 degree -> there is new electric path"""
        angles = []
        n = len(vectors)
        for i in range(n):
            for j in range(i+1, n):
                angle = self.angle_between_vectors(vectors[i], vectors[j])
                angles.append(angle)
        return any(a > threshold for a in angles) or len(vectors) < 2
    
        
    def extend_disconnected_wires(self, optimized_connections, tracking_mask, tolerance=7, max_track_len=2000, radius=20, step=10, turn_threshold=60):
        """Found and connect wires based on new direct"""
        conn_counter = defaultdict(int)
        point_directions = defaultdict(list)
        optimized_connections = set(optimized_connections)
        for p1, p2 in optimized_connections:
            conn_counter[p1] += 1
            conn_counter[p2] += 1

            v = np.array(p2) - np.array(p1)
            if np.linalg.norm(v) == 0: 
                continue
            v_unit = v / np.linalg.norm(v)
            point_directions[p1].append(v_unit)
            point_directions[p2].append(-v_unit)

        candidate_points = []
        for p, count in conn_counter.items():
            if count < 4 and self.has_missing_direction(point_directions[p]):
                candidate_points.append(p)

        new_conns = set()
        for p in candidate_points:
            dir_new = self.get_initial_direction_mask(p, tracking_mask, radius=radius)
            if dir_new:
                q = self.track_mask(p, dir_new, tracking_mask, max_length=max_track_len, step=step, turn_threshold=turn_threshold)
                if np.linalg.norm(np.array(p) - np.array(q)) > 5:
                    dist = np.linalg.norm(np.array(p) - np.array(q))
                    print(f"[track] {p} → {q} | dist = {dist:.2f}")
                    key = tuple(sorted((p, q)))
                    if key not in optimized_connections:
                        new_conns.add(key)

        if new_conns:
            print(f"[+] Added {len(new_conns)} missing wire lines.")
        optimized_connections.update(new_conns)
        return optimized_connections, new_conns
