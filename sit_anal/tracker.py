import numpy as np

class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 1
        self.objects = {}  # id -> (x,y)
        self.max_distance = max_distance

    def update(self, detections):
        centroids = np.array([d['centroid'] for d in detections], dtype=float) if detections else np.zeros((0,2))
        ids = list(self.objects.keys())
        old = np.array([self.objects[i] for i in ids], dtype=float) if ids else np.zeros((0,2))

        assigned = {}
        used_new = set()

        if len(old) and len(centroids):
            dist = ((old[:,None,:]-centroids[None,:,:])**2).sum(axis=2)**0.5
            for oi in np.argsort(dist, axis=None):
                r = oi // dist.shape[1]
                c = oi % dist.shape[1]
                if r in assigned or c in used_new:
                    continue
                if dist[r,c] <= self.max_distance:
                    assigned[r] = c
                    used_new.add(c)

        # update existing
        for idx_pos, obj_id in enumerate(ids):
            if idx_pos in assigned:
                cidx = assigned[idx_pos]
                self.objects[obj_id] = centroids[cidx].tolist()
                detections[cidx]['track_id'] = obj_id
            else:
                # lost -> remove
                del self.objects[obj_id]

        # new objects
        for i in range(len(centroids)):
            if i not in used_new:
                obj_id = self.next_id
                self.next_id += 1
                self.objects[obj_id] = centroids[i].tolist()
                detections[i]['track_id'] = obj_id

        return detections
