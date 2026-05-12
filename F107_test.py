# =========================================================================
#   (c) Copyright 2025
#   All rights reserved
#   Programs written by Zhenduo Wang
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================

from utils import *

# =========================================================
# Settings
# =========================================================

EVAL_DAYS = [1, 27, 45, 60]
DATA_NAME = "F10.7"
SEQ_LEN = 30
BATCH_SIZE = 32

SHARED_PREPROCESS_DIR = "./data"
MODEL_DIR = "./models"

PREPROCESS_PATH = os.path.join(
    SHARED_PREPROCESS_DIR,
    f"{DATA_NAME}_preprocess.pkl"
)

RAW_TEST_CSV_PATH = os.path.join(
    SHARED_PREPROCESS_DIR,
    f"{DATA_NAME}_test.csv"
)

# =========================================================
# Sequence
# =========================================================

def create_sequences_fix(data, timesteps, predict_day):
    X, y = [], []

    for i in range(len(data) - timesteps - predict_day):
        X.append(data.iloc[i:(i + timesteps)])
        y.append(data.iloc[(i + timesteps):(i + timesteps + predict_day)])

    return np.array(X), np.array(y)

def load_and_process_raw_test(
        preprocess_method_path,
        raw_test_csv_path,
        seq_len,
        pred_len,
        test_start="2009-01-01",
        test_end_year=2021
):
    """
    IMPORTANT:
    The same preprocessing file is used for all forecast horizons.
    The scaler is shared.

    But pred_len must be different for day=1, 27, 45, 60.

    Therefore, do NOT use predict_day saved in the pkl.
    Use the pred_len passed from the current loop.
    """

    with open(preprocess_method_path, "rb") as f:
        preprocess_obj = pickle.load(f)

    scaler = preprocess_obj["scaler"]
    value_col = preprocess_obj["value_col"]

    raw_test_df = pd.read_csv(raw_test_csv_path).fillna(0)
    raw_test_df["date"] = pd.to_datetime(raw_test_df["date"])
    raw_test_df = raw_test_df.sort_values("date").reset_index(drop=True)

    # Dynamically cut test range for each prediction length
    start_date = pd.Timestamp(test_start) - pd.DateOffset(
        days=seq_len + pred_len
    )

    test_df = raw_test_df[
        (raw_test_df["date"] > start_date) &
        (raw_test_df["date"].dt.year <= test_end_year)
    ].copy()

    test_scaled = scaler.transform(
        test_df[value_col].values.reshape(-1, 1)
    ).flatten()

    TestX, Testy = create_sequences_fix(
        pd.Series(test_scaled),
        seq_len,
        pred_len
    )

    TestX = TestX.reshape(TestX.shape[0], TestX.shape[1], 1)
    Testy = Testy.reshape(Testy.shape[0], Testy.shape[1], 1)

    return TestX, Testy, scaler, test_df


# =========================================================
# Dataloader
# =========================================================

def build_dataloader(X, y, batch_size=32, shuffle=False):
    dataset = TensorDataset(
        torch.Tensor(X),
        torch.Tensor(y)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


# =========================================================
# Load model
# =========================================================

def load_sinet_model(model_name, configs, device):
    """
    Supports:
    1. torch.save(model, path)
    2. torch.save(model.state_dict(), path)
    """

    ckpt = torch.load(model_name, map_location=device)

    if isinstance(ckpt, dict):
        model = Model(configs).to(device)
        model.load_state_dict(ckpt)
    else:
        model = ckpt.to(device)

    model.eval()
    return model

# =========================================================
# Evaluation
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if not os.path.exists(PREPROCESS_PATH):
    raise FileNotFoundError(f"Missing preprocess file: {PREPROCESS_PATH}")

if not os.path.exists(RAW_TEST_CSV_PATH):
    raise FileNotFoundError(f"Missing raw test csv: {RAW_TEST_CSV_PATH}")

day_results = {
    day: {"rmse": [], "mae": [], "mape": []}
    for day in EVAL_DAYS
}

for day in EVAL_DAYS:
    model_name = os.path.join(
        MODEL_DIR,
        f"SINet_{day}_interval_F10.pth"
    )

    seq_len = SEQ_LEN
    pred_len = day

    print("=" * 80)
    print(f"Data: {DATA_NAME}")
    print(f"Sequence Length: {seq_len}")
    print(f"Predict Length : {pred_len}")
    print(f"Loading model  : {model_name}")

    TestX, Testy, scaler, test_df = load_and_process_raw_test(
        preprocess_method_path=PREPROCESS_PATH,
        raw_test_csv_path=RAW_TEST_CSV_PATH,
        seq_len=seq_len,
        pred_len=pred_len,
        test_start="2009-01-01",
        test_end_year=2021
    )

    test_loader = build_dataloader(
        TestX,
        Testy,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    sl = seq_len
    pl = pred_len

    class Config:
        seq_len = sl
        pred_len = pl
        enc_in = 1
        c_out = 1
        d_model = 32
        d_ff = 64
        num_kernels = 6
        e_layers = 2
        top_k = 3
        task_name = "long_term_forecast"
        dropout = 0.3
        label_len = 1
        freq = "d"
        embed = "timeF"

    configs = Config()

    model = load_sinet_model(
        model_name=model_name,
        configs=configs,
        device=device
    )

    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            outputs = model(x_test, None, None, None)

            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(y_test.cpu().numpy())

    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    test_predictions_original = scaler.inverse_transform(
        test_predictions.reshape(-1, 1)
    ).reshape(test_predictions.shape)

    test_targets_original = scaler.inverse_transform(
        test_targets.reshape(-1, 1)
    ).reshape(test_targets.shape)

    # Evaluate the last forecast step
    day_pred_last = test_predictions_original[:, -1]
    day_true_last = test_targets_original[:, -1]

    rmse = mean_squared_error(
        day_true_last,
        day_pred_last,
        squared=False
    )

    mae = mean_absolute_error(
        day_true_last,
        day_pred_last
    )

    mape = mean_absolute_percentage_error(
        day_true_last,
        day_pred_last
    ) * 100

    day_results[day]["rmse"].append(rmse)
    day_results[day]["mae"].append(mae)
    day_results[day]["mape"].append(mape)

    print(f"Day {day} result:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

print("\n" + "=" * 80)
print("Final Results")
print("=" * 80)

for day in EVAL_DAYS:
    print(
        f"Day {day} - "
        f"RMSE: {np.mean(day_results[day]['rmse']):.4f}, "
        f"MAE: {np.mean(day_results[day]['mae']):.4f}, "
        f"MAPE: {np.mean(day_results[day]['mape']):.2f}%"
    )