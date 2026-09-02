import numpy as np
import torch
from opendataval.dataval.margcontrib.datashap import DataShapley
from opendataval.dataval.dvrl.dvrl import DVRL
from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.experiment.exper_methods import noisy_detection
import pandas as pd

class DVRLShap(DataEvaluator, ModelMixin):
    """
    DVRLShap: A hybrid data valuation method combining Data Shapley and DVRL.
    """

    def __init__(self, dvrl_kwargs=None, datashap_kwargs=None):
        super().__init__()
        self.dvrl_kwargs = dvrl_kwargs or {}
        self.datashap_kwargs = datashap_kwargs or {}
        self.data_values = None
        self._predictor_set = False

    def set_predictor(self, model,metric=None):
        print("[DVRLShap] set_predictor called.")
        self.pred_model = model
        self._predictor_set = True
        self.metric = metric
        print("metric set to:", self.metric)
        print(f"[DVRLShap] self.pred_model set: {type(self.pred_model)}")

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        *args, fetcher=None, **kwargs
    ):
        print("[DVRLShap] input_data called.")
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.fetcher = fetcher  # Store fetcher for noisy detection

        # Ensure pred_model is set before this!
        if hasattr(self, "pred_model"):
            with torch.no_grad():
                y_pred = self.pred_model.predict(self.x_train)
                self.y_pred_diff = (y_pred - self.y_train).float()  # or your logic
        else:
            self.y_pred_diff = torch.zeros_like(self.y_train, dtype=torch.float32)

        self.datashap = DataShapley(**self.datashap_kwargs)
        self.dvrl = DVRL(**self.dvrl_kwargs)

        self.datashap.input_data(x_train, y_train, x_valid, y_valid, *args, **kwargs)
        self.dvrl.input_data(x_train, y_train, x_valid, y_valid, *args, **kwargs)

        if self._predictor_set:
            print("[DVRLShap] Setting predictor for DataShapley and DVRL.")
            self.datashap.pred_model = self.pred_model  # ensure compatibility
            self.dvrl.pred_model = self.pred_model       # ensure compatibility
            self.dvrl.metric = self.metric
            self.datashap.metric = self.metric

        print("[DVRLShap] input_data completed.\n")
        return self

    def train_data_values(self, *args, **kwargs):
        print("[DVRLShap] Training Data Shapley...")
        self.datashap.train_data_values(*args, **kwargs)
        shapley_values = self.datashap.evaluate_data_values()
        print(f"[DVRLShap] Shapley values computed. Example: {shapley_values[:5]}")

        print("[DEBUG] fetcher exists?", hasattr(self, "fetcher"))
        if hasattr(self, "fetcher"):
            print("[DEBUG] type(self.fetcher):", type(self.fetcher))
            print("[DEBUG] hasattr(fetcher, 'noisy_train_indices'):", hasattr(self.fetcher, "noisy_train_indices"))
            print("[DEBUG] fetcher.noisy_train_indices:", getattr(self.fetcher, "noisy_train_indices", None))
            
        # Print noisy detection performance for DataShapley
        shapley_noisy_perf = noisy_detection(self.datashap, fetcher=self.fetcher)
        print(f"[DVRLShap] DataShapley noisy detection F1: {shapley_noisy_perf['kmeans_f1']:.4f}")

        print("[DVRLShap] Initializing DVRL with Shapley values...")
        if hasattr(self.dvrl, "set_initial_values"):
            self.dvrl.set_initial_values(shapley_values)
            print("[DVRLShap] DVRL initialized with Shapley values.")
        else:
            print("[DVRLShap] Warning: DVRL does not support set_initial_values.")

        print("[DVRLShap] Training DVRL...")
        self.dvrl.train_data_values(*args, **kwargs)
        dvrl_values = self.dvrl.evaluate_data_values()
        print(f"[DVRLShap] DVRL values computed. Example: {dvrl_values[:5]}")

        # Print noisy detection performance for DVRL
        self.dvrl.data_values = dvrl_values  # Ensure correct values are used
        dvrl_noisy_perf = noisy_detection(self.dvrl, fetcher=self.fetcher)
        print(f"[DVRLShap] DVRL noisy detection F1: {dvrl_noisy_perf['kmeans_f1']:.4f}")

        print("[DVRLShap] Comparing validation performance...")

        # DEBUG: check model assignments
        print(f"[DEBUG] hasattr(self.datashap, 'pred_model')? {hasattr(self.datashap, 'pred_model')}")
        print(f"[DEBUG] self.datashap.pred_model type: {type(self.datashap.pred_model) if hasattr(self.datashap, 'pred_model') else None}")
        print(f"[DEBUG] self.dvrl.pred_model type: {type(self.dvrl.pred_model) if hasattr(self.dvrl, 'pred_model') else None}")
        print(f"[DEBUG] self.pred_model type: {type(self.pred_model) if hasattr(self, 'pred_model') else None}")

        # Ensure no crash
        if not hasattr(self.datashap, "pred_model"):
            print("[DVRLShap] WARNING: DataShapley missing pred_model — patching.")
            self.datashap.pred_model = self.pred_model
        if not hasattr(self.dvrl, "pred_model"):
            print("[DVRLShap] WARNING: DVRL missing pred_model — patching.")
            self.dvrl.pred_model = self.pred_model

        shapley_perf = self.datashap.evaluate(self.y_valid, self.datashap.pred_model.predict(self.x_valid))
        dvrl_perf = self.dvrl.evaluate(self.y_valid, self.dvrl.pred_model.predict(self.x_valid))

        print(f"[DVRLShap] Shapley performance: {shapley_perf}")
        print(f"[DVRLShap] DVRL performance: {dvrl_perf}")

        if dvrl_perf > shapley_perf:
            print("[DVRLShap] Using DVRL values (better validation performance).")
            self.data_values = dvrl_values
        else:
            print("[DVRLShap] Using Shapley values (better or equal validation performance).")
            self.data_values = shapley_values
        train_indices = np.array(self.fetcher.train_indices)  # global train indices

        # Build DataFrame
        df = pd.DataFrame({
            "index": train_indices,
            "shapley_value": shapley_values,
            "dvrl_value": dvrl_values
        })

        # Sort by index to match global dataset if needed
        df = df.sort_values("index").reset_index(drop=True)

        # Save
        csv_path = "dvrlshap_values.csv"
        df.to_csv(csv_path, index=False)
        print(f"[DVRLShap] Saved Shapley and DVRL values to {csv_path}")
        return self

    def evaluate_data_values(self):
        print("[DVRLShap] Returning final data values.")
        return self.data_values
