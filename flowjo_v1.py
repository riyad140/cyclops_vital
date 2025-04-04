# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:22:27 2024

@author: imrul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from FlowCytometryTools import FCMeasurement
from matplotlib.patches import Polygon
import os


# Load the FCS file
def load_fcs(file_path):
    sample = FCMeasurement(ID='Sample', datafile=file_path)
    data = sample.data
    return data

# Interactive gating function


def manual_gate(data, x_channel, y_channel, gate_label):
    fig, ax = plt.subplots()
    ax.scatter(data[x_channel], data[y_channel], s=1, c='blue', alpha=0.5)
    ax.set_xlabel(x_channel)
    ax.set_ylabel(y_channel)
    ax.set_title(f'Draw Gate: {gate_label}. Close the plot when done.')
    selected_indices = []

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        nonlocal selected_indices
        selected_indices = data[
            (data[x_channel] >= min_x) & (data[x_channel] <= max_x) &
            (data[y_channel] >= min_y) & (data[y_channel] <= max_y)
        ].index.tolist()
        print(f"Selected {len(selected_indices)} events for gate '{gate_label}'.")

    toggle_selector = RectangleSelector(
        ax, onselect, drawtype='box', useblit=True,
        button=[1], minspanx=5, minspany=5, spancoords='pixels',
        interactive=True
    )
    plt.show()
    return selected_indices


from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path


def manual_gate_polygon(data, x_channel, y_channel, gate_label):
    fig, ax = plt.subplots()
    ax.scatter(data[x_channel], data[y_channel], s=1, c='blue', alpha=0.5)
    ax.set_xlabel(x_channel)
    ax.set_ylabel(y_channel)
    ax.set_title(f'Draw Gate: {gate_label}. Close the plot when done.')
    selected_indices = []

    # Function to check if points are inside the polygon
    def onselect(verts):
        # Convert the vertices of the polygon to a path
        path = Path(verts)  # Use Path from matplotlib.path
        
        nonlocal selected_indices
        # Check which points are inside the polygon and get their indices
        selected_indices = data[path.contains_points(list(zip(data[x_channel], data[y_channel])))].index.tolist()
        print(f"Selected {len(selected_indices)} events for gate '{gate_label}'.")

    # Create the polygon selector
    toggle_selector = PolygonSelector(ax, onselect, useblit=True)
    plt.show()

    return selected_indices

# Main gating function (unchanged)
def explicit_gating(data, x_channel, y_channel, gate_labels):
    labeled_data = data.copy()
    labeled_data['Label'] = gate_labels[-1]  # Default label for unassigned events
    remaining_indices = labeled_data.index.tolist()  # Track the unselected indices

    for gate_label in gate_labels[:-1]:
        print(f"Creating gate: {gate_label}")
        
        # Select data that has not been gated yet (unlabeled points)
        remaining_data = labeled_data.loc[remaining_indices]

        # Perform manual polygon gating
        selected_indices = manual_gate_polygon(remaining_data, x_channel, y_channel, gate_label)

        # Check if selected_indices is non-empty (i.e., some points were selected)
        if selected_indices:
            # Label the selected data for this gate
            labeled_data.loc[selected_indices, 'Label'] = gate_label
            print(f"Selected {len(selected_indices)} events for gate '{gate_label}'.")

            # Update the remaining indices by removing the selected points
            remaining_indices = list(set(remaining_indices) - set(selected_indices))

    return labeled_data

# Function to manually gate using a user-defined four-sided polygon
# def manual_gate(data, x_channel, y_channel, gate_label):
#     fig, ax = plt.subplots()
#     ax.scatter(data[x_channel], data[y_channel], s=1, c='blue', alpha=0.5)
#     ax.set_xlabel(x_channel)
#     ax.set_ylabel(y_channel)
#     ax.set_title(f'Draw Gate: {gate_label}. Click four points to define a polygon.')
    
#     selected_indices = []
#     polygon_points = []  # List to store the four points clicked by the user

#     # Function to handle user clicks
#     def on_click(event):
#         if len(polygon_points) < 4:
#             x, y = event.xdata, event.ydata
#             polygon_points.append((x, y))
#             ax.plot(x, y, 'ro')  # Mark the point with a red circle
#             if len(polygon_points) == 4:
#                 # After 4 points, draw the polygon
#                 poly = Polygon(polygon_points, closed=True, edgecolor='r', facecolor='none')
#                 ax.add_patch(poly)
#                 plt.draw()
#                 print("Polygon created with 4 points.")

#     # Function to finalize selection
#     def finalize_selection(event):
#         nonlocal selected_indices
#         # Create a polygon from the selected points
#         polygon = Polygon(polygon_points, closed=True, edgecolor='r', facecolor='none')
#         for idx, (x, y) in enumerate(zip(data[x_channel], data[y_channel])):
#             if polygon.contains_point((x, y)):
#                 selected_indices.append(idx)
#         print(f"Selected {len(selected_indices)} events for gate '{gate_label}'.")

#     # Connect the click event to the on_click function
#     cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
#     cid_finalize = fig.canvas.mpl_connect('key_press_event', finalize_selection)
    
#     plt.show()
    # return selected_indices
#




# Label data based on selected indices
def label_data(data, selected_indices, label):
    labeled_data = data.copy()
    labeled_data['Label'] = np.nan
    labeled_data.loc[selected_indices, 'Label'] = label
    return labeled_data

# Perform explicit gating with three gates and assign the rest
# def explicit_gating(data, x_channel, y_channel, gate_labels):
#     labeled_data = data.copy()
#     labeled_data['Label'] = 'Neutro'  # Default label for unassigned events
#     remaining_indices = labeled_data.index.tolist()

#     for gate_label in gate_labels:
#         print(f"Creating gate: {gate_label}")
#         selected_indices = manual_gate_polygon(labeled_data.loc[remaining_indices], x_channel, y_channel, gate_label)

#         if selected_indices:
#             # Label the selected data
#             labeled_data.loc[selected_indices, 'Label'] = gate_label
#             # Remove labeled indices from the remaining population
#             remaining_indices = list(set(remaining_indices) - set(selected_indices))

#     return labeled_data


def plot_labeled_populations(data, x_channel, y_channel, gate_labels, save_path):
    # Create a scatter plot of the entire dataset with labels
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8.4) 

    # Define a set of colors for different populations
    label_colors = {
        gate_labels[0]: 'darkred',
        gate_labels[1]: 'orange',
        gate_labels[2]: 'green',
        gate_labels[3]: 'blue',
        gate_labels[4]: 'black' # Color for dim cells/ debris
    }

    # Plot each label with a specific color
    for label, color in label_colors.items():
        # Filter the data for the current label
        labeled_data = data[data['Label'] == label]
        
        # Plot the points for this label
        ax.scatter(labeled_data[x_channel], labeled_data[y_channel], 
                   s=1, c=color, label=label, alpha=0.6)
    
    # Set the plot labels and title

    ax.set_xlabel(x_channel)
    ax.set_ylabel(y_channel)
    ax.set_title('Gated Populations with Labels')
    ax.set_xlim([0,1000])
    ax.set_ylim([0,1000])
    
    # Display legend
    ax.legend(title='Gates')
    ax.axvline(x=350, color='black', linestyle='--', label='Vertical Line')
    ax.axhline(y=250, color='black', linestyle='--', label='Vertical Line')
    
    # Show the plot
    plt.savefig(os.path.join(save_path,'Manual_gating'))
    plt.show()


def manual_gating(fcs_file_path):
    save_path = os.path.split(fcs_file_path)[0]
    sampleID = os.path.split(os.path.split(save_path)[0])[-1]
    final_save_path = os.path.join(save_path,'Manual_gating')
    try:
        os.mkdir(final_save_path)
    except:
        print('Folder already exists')
        pass
    
    data = load_fcs(fcs_file_path)

    # Specify channels for gating
    x_channel = 'intensitymediandarkfield'  # Replace with your x-axis channel
    y_channel = 'meanintensityred'  # Replace with your y-axis channel

    # Specify gate labels
    gate_labels = ['Lymph', 'Mono', 'Eos', 'Neutro', 'Dim']

    # Perform explicit gating
    labeled_data = explicit_gating(data, x_channel, y_channel, gate_labels)

    # Save labeled data to CSV
    labeled_data.to_csv(os.path.join(final_save_path,'LabeledPopulation.csv'), index=False)
    print("Labeled data saved to 'LabeledPopulation.csv'")
    
    plot_labeled_populations(labeled_data, x_channel, y_channel, gate_labels,final_save_path)
    
    labels = pd.Series(labeled_data['Label'])
    
    # Calculate percentages
    label_percentages = labels.value_counts(normalize=True) * 100
    
    print(label_percentages)
    
    differentials_0 = [label_percentages['Lymph'],label_percentages['Neutro'],label_percentages['Mono'],label_percentages['Eos'],label_percentages['Dim']]
    differentials_1 = [label_percentages['Lymph']*(1 + label_percentages['Dim']/100), 
                       label_percentages['Neutro']*(1 + label_percentages['Dim']/100),
                       label_percentages['Mono']*(1 + label_percentages['Dim']/100),
                       label_percentages['Eos']*(1 + label_percentages['Dim']/100)]
    
    # lymph neutro(neutro+dim) mid(eos+mono) Mono Eos
    
    diff_strategy_0 = [differentials_0[0], differentials_0[1] + differentials_0[4], differentials_0[2] + differentials_0[3], differentials_0[2], differentials_0[3]] # dim cell added to neutro
    diff_strategy_1 = [differentials_1[0], differentials_1[1], differentials_1[2] + differentials_1[3], differentials_1[2], differentials_1[3]] # dim cell added proportionately everywhere
                       
    
    df_stats = pd.DataFrame(np.vstack([diff_strategy_0,diff_strategy_1]), columns = ['Lymph', 'Neutro', 'Mid', 'Mono','Eos'])
    
    label_percentages.to_csv(os.path.join(final_save_path,f'wbc_stats_manualGating_{sampleID}.csv'),index = True)
    df_stats.to_csv(os.path.join(final_save_path,f'Final_wbc_stats_manualGating_{sampleID}.csv'),index = True)
    

# Main function
if __name__ == "__main__":
    # Load your FCS file
    # fcs_file_path = r"W:\raspberrypi\photos\Offsite\2024-11-27\s035-A\wbc-results-20241127_152050-3part_fullcrop\s035-fcs.fcs"
    
    
    file_paths= [
        'W:\\raspberrypi\\photos\\PV_2025\\B009\\2025-03-11\\S006_MC\\wbc-results-20250311_130350-3part_flagging_nogating\\S006-fcs.fcs',
         'W:\\raspberrypi\\photos\\PV_2025\\B009\\2025-03-11\\S008_MC\\S008\\wbc-results-20250311_141509-3part_flagging_nogating\\S008-fcs.fcs',
         'W:\\raspberrypi\\photos\\PV_2025\\B009\\2025-03-11\\S025_MC\\wbc-results-20250311_144727-3part_flagging_nogating\\S025_MC-fcs.fcs',
         'W:\\raspberrypi\\photos\\PV_2025\\B009\\2025-03-11\\S048_MC\\wbc-results-20250311_131518-3part_flagging_nogating\\S048-fcs.fcs'
        ]
    
    for fcs_file_path in file_paths:
        print(fcs_file_path)
    
        manual_gating(fcs_file_path)
    
    
    # save_path = os.path.split(fcs_file_path)[0]
    # sampleID = os.path.split(os.path.split(save_path)[0])[-1]
    # final_save_path = os.path.join(save_path,'Manual_gating')
    # try:
    #     os.mkdir(final_save_path)
    # except:
    #     print('Folder already exists')
    #     pass
    
    # data = load_fcs(fcs_file_path)

    # # Specify channels for gating
    # x_channel = 'intensitymediandarkfield'  # Replace with your x-axis channel
    # y_channel = 'meanintensityred'  # Replace with your y-axis channel

    # # Specify gate labels
    # gate_labels = ['Lymph', 'Mono', 'Eos', 'Neutro']

    # # Perform explicit gating
    # labeled_data = explicit_gating(data, x_channel, y_channel, gate_labels)

    # # Save labeled data to CSV
    # labeled_data.to_csv(os.path.join(final_save_path,'LabeledPopulation.csv'), index=False)
    # print("Labeled data saved to 'LabeledPopulation.csv'")
    
    # plot_labeled_populations(labeled_data, x_channel, y_channel, gate_labels,final_save_path)
    
    # labels = pd.Series(labeled_data['Label'])
    
    # # Calculate percentages
    # label_percentages = labels.value_counts(normalize=True) * 100
    
    # print(label_percentages)
    # label_percentages.to_csv(os.path.join(final_save_path,f'wbc_stats_manualGating_{sampleID}.csv'),index = True)