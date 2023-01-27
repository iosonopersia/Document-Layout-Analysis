import csv
import json

import matplotlib.pyplot as plt
import numpy as np


class YOLO_KMeans:
    def __init__(self, filename):
        # Load COCO dataset
        with open(filename, "r", encoding="utf-8") as f:
            coco_dataset = json.load(f)

        # Filter images by doc_category
        images_set = {
            image['id']
            for image in coco_dataset['images']
            if image['doc_category'] in ['scientific_articles']
        }

        # Extract anchors from annotations
        boxes = [
            (annotation['bbox'][2], annotation['bbox'][3])
            for annotation in coco_dataset['annotations']
            if annotation['image_id'] in images_set
        ]

        # Normalize each dimension by 1025 (image size 1025x1025)
        self.boxes = np.array(boxes, dtype=np.float64) / 1025

    def _iou_distance(self, clusters):
        """
        Calculate the distance between N boxes and K clusters
        :param clusters: (K, 2) array of cluster centers
        :return: (N, K) array of distances
        """
        # Calculate the area of each box
        box_area = self.boxes[:, 0] * self.boxes[:, 1] # (N,)
        box_area = np.expand_dims(box_area, axis=1) # (N, 1)

        # Calculate the area of each cluster
        cluster_area = clusters[:, 0] * clusters[:, 1] # (K,)
        cluster_area = np.expand_dims(cluster_area, axis=0) # (1, K)

        # Calculate the intersection area between each box and each cluster
        box_w = np.expand_dims(self.boxes[:, 0], axis=1) # (N, 1)
        cluster_w = np.expand_dims(clusters[:, 0], axis=0) # (1, K)
        min_w = np.minimum(cluster_w, box_w) # (N, K) element-wise minimum

        box_h = np.expand_dims(self.boxes[:, 1], axis=1) # (N, 1)
        cluster_h = np.expand_dims(clusters[:, 1], axis=0) # (1, K)
        min_h = np.minimum(cluster_h, box_h) # (N, K) element-wise minimum

        intersection_area = np.multiply(min_w, min_h) # (N, K) element-wise multiplication

        # Calculate the union area between each box and each cluster
        union_area = (box_area + cluster_area) - intersection_area # (N, K) element-wise addition

        result = intersection_area / union_area # (N, K) element-wise division
        return result

    def avg_iou(self, clusters):
        accuracy = np.mean([np.max(self._iou_distance(clusters), axis=1)])
        return accuracy

    def fit(self, k, dist=np.median):
        """
        Runs k-means on the dataset
        :param k: number of clusters
        :param dist: distance function to use (default: np.median)
        :return: (k, 2) array of cluster centers
        """
        # Sanity check
        assert k > 0, "k must be positive"
        k = int(k)

        n = self.boxes.shape[0]
        distances = np.empty((n, k), dtype=np.float64)
        last_nearest = np.zeros((n,), dtype=np.int64)

        # Randomly choose the initial k clusters
        random_state = 42
        rng=np.random.RandomState(random_state)
        clusters = self.boxes[rng.choice(n, k, replace=False)]

        while True:
            distances = 1 - self._iou_distance(clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster_idx in range(k):
                # Update one cluster at a time by taking the median
                # of all points currently assigned to that cluster
                clusters[cluster_idx] = dist(
                    self.boxes[current_nearest == cluster_idx],
                    axis=0
                )

            last_nearest = current_nearest

        return clusters

    def transform(self, clusters):
        """
        Transforms each box to its closest cluster ID
        :return: (N, 2) array of boxes
        """
        distances = 1 - self._iou_distance(clusters)
        return np.argmin(distances, axis=1)

    def save_plot(self, clusters, labels, filename="clusters.png"):
        """
        Plots the clusters and their labels
        :param clusters: (K, 2) array of cluster centers
        :param labels: (N,) array of cluster labels
        """
        plt.figure(figsize=(6, 6))

        plt.scatter(self.boxes[:, 0], self.boxes[:, 1], marker=".", c=labels)
        plt.scatter(clusters[:, 0], clusters[:, 1], marker="x", color="black")

        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.title(f"K-means clustering with k={clusters.shape[0]}")
        plt.xlabel("Width (normalized)")
        plt.ylabel("Height (normalized)")
        plt.legend(["Boxes", "Clusters"])
        plt.savefig(filename)

    def clusters_to_csv(self, clusters, filename="clusters.csv", scale=1):
        """
        Converts the clusters to a CSV file
        :param clusters: (K, 2) array of cluster centers
        :param filename: name of the CSV file (default: clusters.csv)
        :param scale: scale factor to apply to the clusters (default: 1)
        """
        with open(filename, "wt", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["width", "height"])
            writer.writerows(clusters * scale)


if __name__ == "__main__":
    kmeans = YOLO_KMeans("E:/DocLayNet_core/COCO/train.json")
    clusters = kmeans.fit(k=12) # default distance function is np.median, with np.mean AVG_IOU is 2.6% lower
    labels = kmeans.transform(clusters)

    print(f"K anchors:\n {clusters*1025}")

    avg_iou = kmeans.avg_iou(clusters)
    print(f"Avg. IoU: {round(avg_iou * 100.0, 2)}%")

    # Save results
    kmeans.save_plot(clusters, labels, filename="anchors.png")
    kmeans.clusters_to_csv(clusters, filename="anchors.csv")
