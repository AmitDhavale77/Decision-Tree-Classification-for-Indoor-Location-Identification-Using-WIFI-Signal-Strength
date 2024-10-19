import matplotlib.pyplot as plt
import numpy as np
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def tree_to_json(decision_tree, output_file):

    with open(output_file, 'w') as fobj:
        json.dump(decision_tree, fobj, cls=NpEncoder)


def json_to_tree(input_file):

    with open(input_file, 'r') as fobj:
        decision_tree = json.load(fobj)
    return decision_tree


def plot_tree(node, x, y, x_offset, y_offset, ax, depth, max_depth):
    """ Recursively plot the decision tree. """
    # Adjust text size and box size based on depth
    font_size = max(12 - depth, 6)  # Minimum font size 6
    box_style = dict(facecolor='peru', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.9)

    # If it's a leaf node, plot the value
    if node['feature'] is None:
        ax.text(x, y, f"Leaf\n{node['value']}", ha='center', va='center', fontsize=font_size,
                bbox=dict(facecolor='olivedrab', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.9))
        return

    # Plot the decision node
    ax.text(x, y, f"X[{node['feature']}] <= {node['value']}", ha='center', va='center', fontsize=font_size, bbox=box_style)

    # Calculate new positions for the left and right children
    # Scale x_offset by a larger factor for deeper trees to avoid clutter
    horizontal_scale = 2 ** (max_depth - depth)  # Increased spacing with depth
    left_x = x - x_offset * horizontal_scale
    right_x = x + x_offset * horizontal_scale
    next_y = y - y_offset

    # Plot left child (recursively)
    if 'left' in node:
        ax.plot([x, left_x], [y, next_y], 'saddlebrown', lw=1.5)  # Draw the line
        plot_tree(node['left'], left_x, next_y, x_offset, y_offset, ax, depth + 1, max_depth)

    # Plot right child (recursively)
    if 'right' in node:
        ax.plot([x, right_x], [y, next_y], 'saddlebrown', lw=1.5)  # Draw the line
        plot_tree(node['right'], right_x, next_y, x_offset, y_offset, ax, depth + 1, max_depth)


# Main function to visualize a decision tree
def visualize_tree(tree, max_depth):
    x_offset = 0.5  # Base horizontal spacing (adjusted dynamically)
    y_offset = 1  # Fixed vertical spacing

    fig_width = 2 ** max_depth  # Adjust figure width dynamically for large trees
    fig_height = (max_depth + 2) * y_offset  # Set figure height based on depth

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()  # Turn off the axis

    # Call the plotting function with initial positions
    plot_tree(tree, x=0.5 * fig_width, y=max_depth + 1, x_offset=x_offset, y_offset=y_offset, ax=ax, depth=0, max_depth=max_depth)

    plt.show()


if __name__ == "__main__":
    decision_tree = json_to_tree('tree.json')

    tree_depth = 11
    visualize_tree(decision_tree, tree_depth)
