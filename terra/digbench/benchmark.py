import csv
from typing import Dict

import numpy as np


def compute_scores(workspace_image: np.array, path_image_frame: np.array, num_local_workspaces:
int, radial_workspace_dim: float, resolution: float):
    """
    This function computes the scores for the benchmark.
    The main metrics are:
        - success (bool): whether the global path was successful
        - path_length (int): length of the global path
        - num of workspaces (int): number of workspaces visited
        - coverage_area (float): area covered by robot as a fraction of the total area
        
    :param workspace_image: image of the workspace in [0, 1], dig areas are marked in (1., 1., 1.)
    :param path_image_frame: global path in image frame [pixels]
    :param workspaces: workspaces 
    :param radial_workspace_dim: radial dimensions of the workspace [m]
    :param resolution: resolution of the image [m/pixel]
    :return: dictionary of scores
    
    """
    path = path_image_frame * resolution
    # if paths are empty then success is false and scores are 0
    if len(path) == 0 or num_local_workspaces == 0:
        return {"success": False, "path_score": 0, "workspace_score": 0, "coverage_area_fraction": 0}

    # get dig area, marked in (1., 1., 1.) in the workspace image * resulution**2
    dig_area = np.sum(np.all(workspace_image == (1, 1, 1), axis=-1)) * resolution ** 2
    covered_area_fraction = compute_covered_area_fraction(workspace_image, path_image_frame,
                                                          radial_workspace_dim,
                                                          resolution)

    # success if arrays are not empty
    success = len(path) > 0 and num_local_workspaces > 0

    # path length approximation
    path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    path_score = path_length / dig_area ** 0.5 if dig_area != 0 else 0

    # number of workspaces
    num_workspaces = num_local_workspaces
    num_workspaces_score = num_workspaces * np.pi ** 2 / 2 * radial_workspace_dim ** 2 / dig_area if dig_area != 0 \
        else 0

    score_dict = {"success": success, "path_score": path_score, "workspace_score": num_workspaces_score,
                  "covered_area_fraction": covered_area_fraction}
    return score_dict


def compute_covered_area_fraction(workspace_image: np.array, path: np.array, workspace_radial_dim: float,
                                  resolution: float):
    """
    This function computes the area covered by the robot as a fraction of the total area.
    The covered area includes all points that lies closer than workspace_radial_dim for each point of the path.
    An area is marked as covered only if it's a designed dig area (1.) in the workspace image.
    Make sure to count only once the points that are covered by multiple points of the path.
    """
    # if workspace image has 3 channels, convert it to 1 channel
    if len(workspace_image.shape) == 3:
        workspace_image = workspace_image[:, :, 0]

    # if in range 0-255, convert to 0-1
    if np.max(workspace_image) > 1:
        workspace_image = workspace_image / 255

    # print("shape path ", path.shape)
    dig_area = np.sum(workspace_image == 1)

    # Initiate mask with the same shape as workspace_image filled with False values
    covered_area_mask = np.full(workspace_image.shape, False)

    # Generate coordinate arrays
    y_coords, x_coords = np.ogrid[:workspace_image.shape[0], :workspace_image.shape[1]]

    for point in path[:, :2]:
        # Compute the distance from the point to each pixel in the workspace_image
        distances = np.sqrt((x_coords - point[0]) ** 2 + (y_coords - point[1]) ** 2)

        # Mark the covered points in the mask
        covered_area_mask |= distances < workspace_radial_dim / resolution
    # Compute covered area by summing over the mask, but only where the workspace_image indicates a dig area
    covered_area = np.sum(covered_area_mask & (workspace_image == 1))
    # plt.imshow(workspace_image, alpha=1)
    # plt.imshow(covered_area_mask, alpha=0.2)
    # plt.show()
    return covered_area / dig_area


def process_scores(case_dict: dict, save_folder: str, resume: bool = False):
    """
    Process the scores for a single case.
    Case dict should contains for each filename:
    - filename: str
        - path: np.ndarray
        - workspacers: np.ndarray
        - scores:
            - success (bool)
            - path_score (float)
            - workspace_score (float)
    The function extracts the scores from the dict and saves them in a CSV file.
    It also creates a file containing the average scores (only using successful cases),
    percentage of success, and the number of cases.
    """

    # Extract the scores from the case_dict for successful cases
    # order the case_dict by filename
    case_dict = dict(sorted(case_dict.items()))
    print("case dict keys are: ", case_dict.keys())
    print("case dict values are: ", case_dict.values())
    scores = []
    for filename, data in case_dict.items():
        success = data['scores']['success']
        path_score = data['scores']['path_score']
        workspace_score = data['scores']['workspace_score']
        covered_area_fraction = data['scores']['covered_area_fraction']
        angle_opt = data['angle_opt']
        scores.append([filename, success, angle_opt, path_score, workspace_score, covered_area_fraction])

    # Save the scores of successful cases in a CSV file
    csv_filename = save_folder + '/successful_scores.csv'
    # if resume is True, append to the existing file
    if resume:
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(scores)
    # if resume is False, create a new file
    else:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ['filename', 'success', 'angle_opt', 'path_score', 'workspace_score', 'covered_area_fraction'])
            writer.writerows(scores)
    # count and exclude from case_dict cases with success = True and coverage_area_fraction = -1
    degenerate_cases = 0
    for filename, data in case_dict.items():
        success = data['scores']['success']
        covered_area_fraction = data['scores']['covered_area_fraction']
        if success and covered_area_fraction == -1:
            degenerate_cases += 1
            del case_dict[filename]
    # Calculate average scores, percentage of success, and number of cases
    num_cases = len(case_dict)
    success_count = sum([score[1] for score in scores])
    success_percentage = (success_count / num_cases) * 100
    successful_scores = [score[2:] for score in scores if score[1]]
    #
    successful_avg_path_score = np.mean([score[0] for score in successful_scores])
    successful_std_path_score = np.std([score[0] for score in successful_scores])
    successful_avg_workspace_score = np.mean([score[1] for score in successful_scores])
    successful_std_workspace_score = np.std([score[1] for score in successful_scores])
    successful_avg_covered_area_fraction = np.mean([score[2] for score in successful_scores])
    successful_std_covered_area_fraction = np.std([score[2] for score in successful_scores])

    # Save average scores, percentage of success, and number of cases in a file
    summary_filename = save_folder + '/summary.txt'
    with open(summary_filename, 'w') as file:
        file.write(f"Number of processed cases: {num_cases}\n")
        file.write(f"Number of incorrect input images: {degenerate_cases}\n")
        file.write(f"Percentage of success: {success_percentage:.2f}%\n")
        file.write(
            f"Mean path scores (std) for successful cases only: {successful_avg_path_score:.2f} ({successful_std_path_score:.2f})\n")
        file.write(
            f"Mean workspace scores (std) for successful cases only: {successful_avg_workspace_score:.2f} ({successful_std_workspace_score:.2f})\n")
        file.write(
            f"Mean covered area fraction (std) for successful cases only: {successful_avg_covered_area_fraction:.2f} ({successful_std_covered_area_fraction:.2f})\n")


def save_score(filename, save_folder: str, score: Dict, angle_opt: float = 0):
    """
    Save a single score value in a CSV file named 'successful_scores.csv'. If it exists append to it.
    """
    csv_filename = save_folder + '/successful_scores.csv'
    # file does not exist add this header ['filename', 'path_score', 'workspace_score', 'covered_area_fraction']
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'angle_opt', 'path_score', 'workspace_score', 'covered_area_fraction'])

    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        try:
            writer.writerow([filename, angle_opt, score['path_score'], score['workspace_score'],
                             score['covered_area_fraction']])
        except KeyError:
            print("score does not contain all the keys")


def process_csv_scores(csv_filename: str):
    """
    This function reads a CSV file containing scores and returns a dictionary with the scores.
    Header: filename, success (bool), angle_opt (float), path_score (float), workspace_score (float),
    covered_area_fraction (float)
    """
    scores = {}
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            filename = row[0]
            success = row[1] == 'True'  # Convert to bool
            angle_opt = float(row[2])  # Convert to float
            path_score = float(row[3])  # Convert to float
            workspace_score = float(row[4])  # Convert to float
            covered_area_fraction = float(row[5])  # Convert to float
            scores[filename] = {'success': success, 'angle_opt': angle_opt, 'path_score': path_score,
                                'workspace_score': workspace_score, 'covered_area_fraction': covered_area_fraction}
    return scores


def summarize_scores(scores: Dict):
    """
    This function takes a dictionary of scores and returns a dictionary with the average scores.
    """
    num_cases = len(scores)
    success_count = sum([score['success'] for score in scores.values()])
    success_percentage = (success_count / num_cases) * 100
    # exclude unsuccessful cases
    successful_scores = {k: v for k, v in scores.items() if v['success']}
    avg_path_score = np.mean([score['path_score'] for score in successful_scores.values()])
    std_path_score = np.std([score['path_score'] for score in successful_scores.values()])
    avg_workspace_score = np.mean([score['workspace_score'] for score in successful_scores.values()])
    std_workspace_score = np.std([score['workspace_score'] for score in successful_scores.values()])
    avg_covered_area_fraction = np.mean([score['covered_area_fraction'] for score in successful_scores.values()])
    std_covered_area_fraction = np.std([score['covered_area_fraction'] for score in successful_scores.values()])
    summary = {'num_cases': num_cases, 'success_percentage': success_percentage, 'avg_path_score': avg_path_score,
               'std_path_score': std_path_score, 'avg_workspace_score': avg_workspace_score,
               'std_workspace_score': std_workspace_score, 'avg_covered_area_fraction': avg_covered_area_fraction,
               'std_covered_area_fraction': std_covered_area_fraction}
    return summary

