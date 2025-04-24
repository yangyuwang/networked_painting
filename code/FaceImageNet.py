import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from retinaface import RetinaFace

class FaceImage:
    def __init__(self, img_path: str, offset: float = 10.0):
        """
        img_path: path to the image file
        offset: pixel offset for drawing arrows (if needed)
        """
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        h, w = self.img.shape[:2]
        self.h, self.w = h, w
        focal_length = w
        self.camera_matrix = np.array([
            [focal_length,       0, w / 2],
            [      0, focal_length, h / 2],
            [      0,           0,     1]
        ], dtype="double")
        self.dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

        # Generic 3D face model (in mm)
        self.model_points = np.array([
            [   0.0,    0.0,    0.0],   # nose tip
            [-30.0, -30.0, -30.0],     # left eye
            [ 30.0, -30.0, -30.0],     # right eye
            [-30.0,  30.0, -30.0],     # left mouth
            [ 30.0,  30.0, -30.0]      # right mouth
        ])
        self.offset = offset

        # To be filled by processing
        self.faces = {}         # id -> raw detection
        self.face_info = []     # list of {id, center, orientation}
        self.G = nx.DiGraph()   # directed graph

        self.__detect_faces()
        self.__compute_pose()
        self.__build_graph()

    def __detect_faces(self):
        """Detect faces with RetinaFace."""
        self.faces = RetinaFace.detect_faces(self.img_path)

    def __compute_pose(self):
        """Estimate head pose and extract orientation vectors."""
        self.face_info.clear()
        for face_id, data in self.faces.items():
            x1, y1, x2, y2 = data['facial_area']
            lm = data['landmarks']
            image_points = np.array([
                lm['nose'],
                lm['left_eye'],
                lm['right_eye'],
                lm['mouth_left'],
                lm['mouth_right']
            ], dtype="double")

            _, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )

            # project forward direction (Z axis) and nose origin
            forward_3d = np.array([[0.0, 0.0, 1000.0]])
            proj_forward, _ = cv2.projectPoints(forward_3d, rvec, tvec,
                                                self.camera_matrix, self.dist_coeffs)
            proj_nose, _    = cv2.projectPoints(self.model_points[0:1], rvec, tvec,
                                                self.camera_matrix, self.dist_coeffs)
            pf = proj_forward.reshape(-1)
            pn = proj_nose.reshape(-1)

            orientation = pf - pn
            orientation = orientation / np.linalg.norm(orientation)
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

            self.face_info.append({'id': face_id,
                                   'center': center,
                                   'orientation': orientation})

    def __build_graph(self):
        """Build directed graph using cosine similarity weights."""
        self.G.clear()
        for f in self.face_info:
            # ensure orientation unit-length
            ori = f['orientation'] / np.linalg.norm(f['orientation'])
            self.G.add_node(f['id'], center=f['center'], orientation=ori)

        for fi in self.face_info:
            for fj in self.face_info:
                if fi['id'] == fj['id']:
                    continue
                vec = fj['center'] - fi['center']
                vec = vec / np.linalg.norm(vec)
                # cosine similarity formula:
                ori = self.G.nodes[fi['id']]['orientation']
                cos_sim = np.dot(ori, vec) / (
                    np.linalg.norm(ori) * np.linalg.norm(vec)
                )
                # clamp within [-1,1]
                cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
                self.G.add_edge(fi['id'], fj['id'], weight=cos_sim)

    def get_faces(self):
        return [
            {'id': fid,
             'bbox': self.faces[fid]['facial_area'],
             'landmarks': self.faces[fid]['landmarks']}
            for fid in self.faces
        ]

    def get_orientation_vectors(self):
        return {f['id']: f['orientation'] for f in self.face_info}

    def get_between_face_vectors(self):
        vecs = {}
        for fi in self.face_info:
            for fj in self.face_info:
                if fi['id'] == fj['id']:
                    continue
                v = fj['center'] - fi['center']
                vecs[(fi['id'], fj['id'])] = v / np.linalg.norm(v)
        return vecs

    def get_network(self):
        return [(u, v, d['weight']) for u, v, d in self.G.edges(data=True)]

    def plot_network(self, output_path: str):
        index_map = {n: i for i, n in enumerate(self.G.nodes())}
        img_out = self.img.copy()
        for u, v, data in self.G.edges(data=True):
            p1 = self.G.nodes[u]['center']
            p2 = self.G.nodes[v]['center']
            sign = 1 if index_map[u] < index_map[v] else -1
            shift = self.offset * sign
            c1 = tuple((p1 + shift).astype(int))
            c2 = tuple((p2 + shift).astype(int))
            cv2.arrowedLine(img_out, c1, c2, (0, 0, 0), 2, tipLength=0.02)
            mid = tuple(((np.array(c1) + np.array(c2)) // 2).astype(int))
            cv2.putText(img_out, f"{data['weight']:.2f}", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.show()
