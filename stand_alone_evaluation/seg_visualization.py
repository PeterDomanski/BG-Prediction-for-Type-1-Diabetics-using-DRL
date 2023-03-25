import csv
import pandas
import argparse
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv_path", dest="csv_path", default=".")
argparser.add_argument("--img_path", dest="img_path", default=".")
argparser.add_argument("--ph", dest="ph", default=30)
argparser.add_argument("--patient", dest="patient", default="")
args = argparser.parse_args()

# ----------------------------------------- Global parameters (config) -------------------------------------------------
csv_path = args.csv_path
img_path = args.img_path
patient_id = args.patient
prediction_horizon = args.ph


# -------------------------------------------- Function definitions ----------------------------------------------------

def load_csv_data(path):
    df = pandas.read_csv(path)
    return df


def data_to_float(values):
    float_values = []
    for v in values:
        row_values = []
        for d in v:
            if d != "":
                row_values.append(float(d))
        float_values.append(row_values)
    return float_values


def plot_surveillance_error_grid(data_frame, img, subject_id):
    fig, ax = plt.subplots()
    img = plt.imread(img)
    ax.imshow(img, extent=[0, 600, 0, 600])
    prediction = [x.strip('[ ]').split(',') for x in data_frame['prediction'].values]
    ground_truth = [x.strip('[ ]').split(',') for x in data_frame['ground_truth'].values]
    for pred, gt in zip(prediction, ground_truth):
        pred_values = [x.strip('[ ]').split(',') for x in pred]
        pred_values = data_to_float([z[0].split(" ") for z in pred_values])
        gt_values = [x.strip('[ ]').split(',') for x in gt]
        gt_values = data_to_float([z[0].split(" ") for z in gt_values])
        plt.scatter(gt_values, pred_values, color="blue", s=1)
    ax.set_xlabel("Measured Blood Glucose Values (ml/dl)")
    ax.set_ylabel("Predicted Blood Glucose Values (ml/dl)")
    plt.title("Surveillance Error Grid for Patient {} using DRL".format(subject_id))
    plt.savefig("./seg_{}min_{}.pdf".format(prediction_horizon, subject_id), dpi=600)


def calculate_seg_risks(data_frame):
    data_points = len(data_frame) * (int(prediction_horizon) / 5)
    risk_levels = {
        'None': 0,
        'Slight': 0,
        'Moderate': 0,
        'Great': 0,
        'Extreme': 0,
    }
    prediction = [x.strip('[ ]').split(',') for x in data_frame['prediction'].values]
    ground_truth = [x.strip('[ ]').split(',') for x in data_frame['ground_truth'].values]
    for pred, gt in zip(prediction, ground_truth):
        pred_values = [x.strip('[ ]').split(',') for x in pred]
        pred_values = data_to_float([z[0].split(" ") for z in pred_values])
        gt_values = [x.strip('[ ]').split(',') for x in gt]
        gt_values = data_to_float([z[0].split(" ") for z in gt_values])
        for p_val, g_val in zip(pred_values, gt_values):
            for p, g in zip(p_val, g_val):
                if 0 <= p <= 120:
                    ground_truth_seg_regions(g, "A", risk_levels)
                elif 120 < p <= 240:
                    ground_truth_seg_regions(g, "B", risk_levels)
                elif 240 < p <= 360:
                    ground_truth_seg_regions(g, "C", risk_levels)
                elif 360 < p <= 480:
                    ground_truth_seg_regions(g, "D", risk_levels)
                elif 480 < p <= 600:
                    ground_truth_seg_regions(g, "E", risk_levels)
    risk_levels_percentage = {k: (v / data_points) * 100 for k, v in risk_levels.items()}
    with open("seg_risk_levels_{}.csv".format(patient_id), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=risk_levels_percentage.keys())
        writer.writeheader()
        writer.writerow(risk_levels_percentage)


def ground_truth_seg_regions(val, level, risk_levels):
    if 0 <= val <= 120:
        if level == "A":
            risk_levels["None"] += 1
        elif level == "B":
            risk_levels['Slight'] += 1
        elif level == "C":
            risk_levels['Moderate'] += 1
        elif level == "D":
            risk_levels['Extreme'] += 1
        elif level == "E":
            risk_levels['Extreme'] += 1
    elif 120 < val <= 240:
        if level == "A":
            risk_levels["Slight"] += 1
        elif level == "B":
            risk_levels["None"] += 1
        elif level == "C":
            risk_levels['Slight'] += 1
        elif level == "D":
            risk_levels['Great'] += 1
        elif level == "E":
            risk_levels['Extreme'] += 1
    elif 240 < val <= 360:
        if level == "A":
            risk_levels["Moderate"] += 1
        elif level == "B":
            risk_levels["Slight"] += 1
        elif level == "C":
            risk_levels["None"] += 1
        elif level == "D":
            risk_levels['Moderate'] += 1
        elif level == "E":
            risk_levels['Great'] += 1
    elif 360 < val <= 480:
        if level == "A":
            risk_levels["Great"] += 1
        elif level == "B":
            risk_levels["Moderate"] += 1
        elif level == "C":
            risk_levels["Slight"] += 1
        elif level == "D":
            risk_levels["None"] += 1
        elif level == "E":
            risk_levels['Moderate'] += 1
    elif 480 < val <= 600:
        if level == "A":
            risk_levels["Extreme"] += 1
        elif level == "B":
            risk_levels["Great"] += 1
        elif level == "C":
            risk_levels["Moderate"] += 1
        elif level == "D":
            risk_levels["Slight"] += 1
        elif level == "E":
            risk_levels["None"] += 1


# ------------------------------------------------- Main loop ----------------------------------------------------------

if __name__ == '__main__':
    data = load_csv_data(csv_path)
    plot_surveillance_error_grid(data, img_path, patient_id)
    calculate_seg_risks(data)
