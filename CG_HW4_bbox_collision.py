import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class BoundingBox:
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax):
        self.xmin, self.ymin, self.zmin = xmin, ymin, zmin
        self.xmax, self.ymax, self.zmax = xmax, ymax, zmax

    def is_colliding(self, other):
        """Check if this bounding box collides with another."""
        return (self.xmin <= other.xmax and self.xmax >= other.xmin and
                self.ymin <= other.ymax and self.ymax >= other.ymin and
                self.zmin <= other.zmax and self.zmax >= other.zmin)



def generate_random_bounding_boxes(n, space_size=10):
    """Generate `n` random bounding boxes within the given space."""
    boxes = []
    for _ in range(n):
        x, y, z = np.random.uniform(0, space_size, 3)
        size = np.random.uniform(1, 3)
        boxes.append(BoundingBox(x, y, z, x + size, y + size, z + size))
    return boxes


''' TODO '''
class BVHNode:
    """Node for the Bounding Volume Hierarchy (BVH) Tree."""
    bbox : int
    lchild : int | None
    rchild : int | None


    def __init__(self, bbox:int, left: int |None = None, right: int | None = None):
        self.bbox = bbox
        self.lchild = left
        self.rchild = right
    
    def add_children(self, left = None, right = None):
        self.lchild = left
        self.rchild = right
        return self
    
    def is_leaf(self,node):
        return node.lchild is None and node.rchild is None



def merge(left: BVHNode, right: BVHNode):
        box_i: int = numleafs
        bounding_boxes.append(
            BoundingBox(
                xmin=min(
                    bounding_boxes[left.bbox].xmin,
                    bounding_boxes[right.bbox].xmin,
                ),
                xmax=max(
                    bounding_boxes[left.bbox].xmax,
                    bounding_boxes[right.bbox].xmax,
                ),
                ymin=min(
                    bounding_boxes[left.bbox].ymin,
                    bounding_boxes[right.bbox].ymin,
                ),
                ymax=max(
                    bounding_boxes[left.bbox].ymax,
                    bounding_boxes[right.bbox].ymax,
                ),
                zmin=min(
                    bounding_boxes[left.bbox].zmin,
                    bounding_boxes[right.bbox].zmin,
                ),
                zmax=max(
                    bounding_boxes[left.bbox].zmax,
                    bounding_boxes[right.bbox].zmax,
                ),
            )
        )
        return BVHNode(bbox=box_i, left=left, right=right)


''' TODO '''
def build_bvh(start,stop):
    """Recursively build a BVH tree."""
    length = stop - start
    if length == 1:
        return BVHNode(bbox=start)
    mid = start + length // 2
    return merge(
            build_bvh(start, mid), build_bvh(mid, stop)
        )


def collide_r(node: BVHNode, index: int):
        if node.bbox == index:
            return False
        if node.bbox < numleafs:
            return bool(
                bounding_boxes[node.bbox].is_colliding(bounding_boxes[index])
            )
        if not bounding_boxes[node.bbox].is_colliding(bounding_boxes[index]):
            return False
        if node.lchild is None or node.rchild is None:
            print("serious error")
            exit(-1)
        return collide_r(node.lchild, index) or collide_r(
            node.rchild, index
        )

''' TODO '''
def detect_collisions_bvh(root):
    """Detect collisions using BVH by comparing all leaf nodes."""
    if root == None:
        return set()
        
    collisions = set()

    for i in range(numleafs):
        if collide_r(root, i):
            collisions.add(i)

    print(f"Collisions Detected: {len(collisions)} pairs")
    for box_i in collisions:
        print(
            f"Collision: Box({bounding_boxes[box_i].xmin},"
            f" {bounding_boxes[box_i].ymin}, {bounding_boxes[box_i].zmin})"
            )
    return collisions


def draw_bounding_box(ax, box, color):
    """Draw a 3D bounding box."""
    vertices = np.array([
        [box.xmin, box.ymin, box.zmin],
        [box.xmax, box.ymin, box.zmin],
        [box.xmax, box.ymax, box.zmin],
        [box.xmin, box.ymax, box.zmin],
        [box.xmin, box.ymin, box.zmax],
        [box.xmax, box.ymin, box.zmax],
        [box.xmax, box.ymax, box.zmax],
        [box.xmin, box.ymax, box.zmax]
    ])
    
    faces = [[vertices[j] for j in [0, 1, 2, 3]], 
             [vertices[j] for j in [4, 5, 6, 7]], 
             [vertices[j] for j in [0, 1, 5, 4]], 
             [vertices[j] for j in [2, 3, 7, 6]], 
             [vertices[j] for j in [1, 2, 6, 5]], 
             [vertices[j] for j in [4, 7, 3, 0]]]

    ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, linewidths=1, edgecolors='k', facecolors=color))


def visualize_bounding_boxes(boxes, collisions):
    """Plot bounding boxes and highlight colliding ones, with coordinates."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract all boxes involved in collisions
    collided_boxes = set()
    for box1, box2 in collisions:
        collided_boxes.add(box1)
        collided_boxes.add(box2)

    # Draw each bounding box with the correct color and add coordinate labels
    for box in boxes:
        color = 'red' if box in collided_boxes else 'blue'
        draw_bounding_box(ax, box, color)

    # Set axis labels
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    plt.title("Bounding Box Collision Visualization (BVH)")
    plt.show()


if __name__ == "__main__":
    num_meshes = 5
    global bounding_boxes
    bounding_boxes = generate_random_bounding_boxes(num_meshes)

    ''' TODO stars '''
    # Step1: Build BVH
    global numleafs
    numleafs = len(bounding_boxes)
    tree = build_bvh(0,numleafs)


    # Step2: Detect collisions using BVH
    collisions = detect_collisions_bvh(tree)  # call function to get this. Now, set it to empty for visualization
    ''' TODO ends '''

    # Visualize the bounding boxes
    visualize_bounding_boxes(bounding_boxes, collisions)