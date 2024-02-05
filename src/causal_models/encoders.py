import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable, Function
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import copy
from typing import Optional, Union, Tuple
from src.utils import windowing

# this is the basic code for the RNN-based encoders used for representation learning in the causal models

# if torch.cuda.is_available():
#     dev = torch.device("cuda")
# else:
#     dev = torch.device("cpu")
# model fits are generally sufficiently fast on CPU  (CUDA support is also not included in conda environment)
dev = torch.device("cpu")


def create_torch_dataset(
    X: pd.DataFrame,
    window_size: int,
    Y: Optional[pd.DataFrame] = None,
    T: Optional[Union[pd.Series, pd.DataFrame]] = None,
    sample_weight: Optional[pd.DataFrame] = None,
    sort_by_date: bool = False,
) -> TensorDataset:
    """Creates torch dataset for training of and prediction with representation encoder models.

    Parameters
    ----------
    X : pd.DataFrame
        Covariate features.
    window_size : int
        The size of fitting windows.
    Y : Optional[pd.DataFrame], optional
        The outcome feature for training, by default None
    T : Optional[pd.DataFrame], optional
        Treatments, by default None
    sample_weight : Optional[pd.DataFrame], optional
        Sample weights for training, by default None
    sort_by_date : bool, optional
        Whether inputs should be sorted by date, by default False

    Returns
    -------
    TensorDataset
        Tensor dataset of windowed X, and optionally Y, T and sample weights.
    """
    X = windowing(X, window_size)

    # sort by date
    if sort_by_date:
        X, Y, T, sample_weight = (
            df.sort_index(level="time_id") if df is not None else None
            for df in [X, Y, T, sample_weight]
        )

    X_time = X.loc[:, X.columns.get_level_values("step") != "static"]
    X_time = X_time.groupby(axis=1, level=0).apply(lambda row: row.values).values
    X_time = np.stack(X_time, axis=-1)

    #  static features will just be copied for all time steps
    X_static = X.loc[:, X.columns.get_level_values("step") == "static"].values
    X_static = np.tile(X_static[:, np.newaxis, :], (1, X_time.shape[1], 1))

    X_all = np.concatenate([X_time, X_static], axis=-1)

    X_tensor = torch.Tensor(X_all).to(dev)

    outputs = [X_tensor]
    if Y is not None:
        outputs.append(torch.Tensor(Y.values).to(dev))
    if T is not None:
        outputs.append(torch.Tensor(T.values).to(dev))
    if sample_weight is not None:
        outputs.append(torch.Tensor(sample_weight).to(dev))

    return TensorDataset(*outputs)


def grad_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    class ReverseGrad(Function):
        """
        Gradient reversal layer
        """

        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            return x

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
            return scale * grad_output.neg()

    return ReverseGrad.apply(x)


class VariationalLSTM(nn.Module):
    """
    Variational LSTM layer
    """

    def __init__(self, input_size, hidden_size, num_layer=1, dropout=0.0):
        super().__init__()

        self.lstm_layers = [nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)]
        if num_layer > 1:
            self.lstm_layers += [
                nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
                for _ in range(num_layer - 1)
            ]
        self.lstm_layers = nn.ModuleList(self.lstm_layers)

        self.hidden_size = hidden_size
        self.dropout = dropout

    def forward(
        self,
        input: torch.Tensor,
        hx_cx: Tuple[torch.Tensor, torch.Tensor],
        variational_inference: bool = False,
    ):
        x = input
        for i, lstm_cell in enumerate(self.lstm_layers):
            hx, cx = hx_cx
            hx, cx = hx[i], cx[i]
            # Variational dropout - sampled once per batch
            out_dropout = torch.bernoulli(
                hx.data.new(hx.data.size()).fill_(1 - self.dropout)
            ) / (1 - self.dropout)
            h_dropout = torch.bernoulli(
                hx.data.new(hx.data.size()).fill_(1 - self.dropout)
            ) / (1 - self.dropout)
            c_dropout = torch.bernoulli(
                cx.data.new(cx.data.size()).fill_(1 - self.dropout)
            ) / (1 - self.dropout)

            output = []
            for t in range(x.shape[1]):
                hx, cx = lstm_cell(x[:, t, :], (hx, cx))
                if lstm_cell.training or variational_inference:
                    out = hx * out_dropout
                    hx, cx = hx * h_dropout, cx * c_dropout
                else:
                    out = hx
                output.append(out)

            x = torch.stack(output, dim=1)

        return x, (hx, cx)


class BRUnit(nn.Module):
    """
    The part of the encoder network that learns the balanced representation with a recurrent NN (LSTM).
    """

    def __init__(
        self,
        input_size: int,
        seq_hidden_size: int,
        br_size: int,
        seq_num_layers: int,
        dropout: float = 0.1,
        input_dropout: float = 0,
    ) -> None:
        super(BRUnit, self).__init__()
        self.br_size = br_size
        self.seq_hidden_size = seq_hidden_size
        self.seq_num_layers = seq_num_layers
        self.input_dropout = input_dropout

        self.lstm = VariationalLSTM(
            input_size,
            seq_hidden_size,
            seq_num_layers,
            dropout=dropout,
        ).to(dev)
        self.linear_br = nn.Linear(seq_hidden_size, br_size).to(dev)
        self.elu = nn.ELU()

    def forward(
        self, X: torch.Tensor, variational_inference: bool = False
    ) -> torch.Tensor:
        h_0 = Variable(
            torch.zeros(self.seq_num_layers, X.size(0), self.seq_hidden_size)
        ).to(dev)
        c_0 = Variable(
            torch.zeros(self.seq_num_layers, X.size(0), self.seq_hidden_size)
        ).to(dev)

        # input dropout: Do not use nn.Dropout because this will set random elements of X to zero;
        # with the mask, some features are set to 0 for all time steps
        if self.training:
            mask = torch.rand(X.size(0), X.size(2)) > self.input_dropout
            mask = mask.unsqueeze(1).expand(-1, X.size(1), -1).to(X.device)
            X[~mask] = 0

        X, _ = self.lstm(X, (h_0, c_0), variational_inference)
        X = X[:, -1, :]
        br = self.linear_br(X)
        br = self.elu(br)
        return br


class OutcomeHead(nn.Module):
    """
    The part of the encoder network that learns to predict the outcome
    based on the balanced representation with a feed-forward NN.
    Should be accessible as y_model.
    """

    def __init__(
        self,
        br_unit: BRUnit,
        fc_hidden_size: int,
        dim_treatments: int,
        output_size: int,
        window_size: int,
        activation_function: str = "linear",
    ) -> None:
        super(OutcomeHead, self).__init__()
        # needs to have access to the br_unit so that it can be used as a y_model for predictions
        self.br_unit = br_unit
        self.window_size = window_size
        self.linear_y1 = nn.Linear(
            self.br_unit.br_size + dim_treatments, fc_hidden_size
        ).to(dev)
        self.elu_y = nn.ELU()
        self.linear_y2 = nn.Linear(fc_hidden_size, output_size).to(dev)
        self.X_mean: pd.Series
        self.X_std: pd.Series
        self.Y_mean: pd.Series
        self.Y_std: pd.Series

        if activation_function == "linear":
            self.activation_function = nn.Identity()
        else:
            raise ValueError(
                f"{activation_function} currently not supported as activation function in outcome head."
            )

    def forward(
        self, br: torch.Tensor, T_current: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the outcome head.

        Parameters
        ----------
        br : torch.Tensor
            The balanced representation.
        T_current : Optional[torch.Tensor]
            Current treatment (potentially more than one dimensions)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Predicted y and balanced representation.
        """
        if T_current is not None:
            if T_current.ndim == 1:
                # if only one treatment: Add second dimension
                T_current = T_current.unsqueeze(-1)
            X = torch.cat((br, T_current), dim=-1)
        else:
            X = br
        X = self.elu_y(self.linear_y1(X))
        Y_pred = self.linear_y2(X)
        Y_pred = self.activation_function(Y_pred)
        return Y_pred, br

    def predict(
        self,
        X: pd.DataFrame,
        T: Optional[Union[pd.Series, pd.DataFrame]] = None,
        variational_inference: bool = False,
    ):
        X = (X - self.X_mean) / self.X_std
        dataset = create_torch_dataset(X, T=T, window_size=self.window_size)
        data_loader = DataLoader(dataset, batch_size=len(X), shuffle=False)

        self.eval()
        with torch.no_grad():
            if T is not None:
                for X_batch, T_batch in data_loader:
                    br_batch = self.br_unit(X_batch, variational_inference)
                    try:
                        outputs, _ = self(br_batch, T_batch)
                    except:
                        outputs, _ = self(br_batch, None)
            else:
                for (X_batch,) in data_loader:
                    br_batch = self.br_unit(X_batch, variational_inference)
                    outputs, _ = self(br_batch, None)

        output = (outputs.cpu().numpy() * pd.Series(self.Y_std).values) + pd.Series(
            self.Y_mean
        ).values
        output = pd.DataFrame(output, index=X.index)
        return output

    def __len__(self):
        return 1


class TreatmentHead(nn.Module):
    """
    The part of the CRN that learns to predict the treatment
    based on the balanced representation with a feed-forward NN.
    Should be accessible as t_model.
    """

    def __init__(
        self,
        br_unit: BRUnit,
        fc_hidden_size: int,
        dim_treatments: int,
        balancing: Optional[str] = "grad_reverse",
        alpha: float = 1.0,
        window_size: int = 7,
        activation_function: str = "sigmoid",
    ):
        super(TreatmentHead, self).__init__()
        # needs to have access to the br_unit so that it can be used as a t_model for predictions
        self.br_unit = br_unit
        self.window_size = window_size
        self.linear_t1 = nn.Linear(self.br_unit.br_size, fc_hidden_size).to(dev)
        self.elu_t = nn.ELU()
        self.linear_t2 = nn.Linear(fc_hidden_size, dim_treatments).to(dev)
        self.X_mean: pd.Series
        self.X_std: pd.Series

        if activation_function == "sigmoid":
            self.activation_function = nn.Sigmoid()
        elif activation_function == "softmax":
            self.activation_function = nn.Softmax()
        else:
            raise ValueError(
                f"{activation_function} currently not supported as activation function in treatment head."
            )

        self.balancing = balancing
        self.alpha = alpha

    def __len__(self):
        return 1

    def forward(self, br: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the treatment head.

        Parameters
        ----------
        br : torch.Tensor
            The balanced representation.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Predicted t and balanced representation.
        """
        if self.balancing == "grad_reverse":
            br = grad_reverse(br, self.alpha)

        br = self.elu_t(self.linear_t1(br))
        T_pred = self.linear_t2(br)
        T_pred = self.activation_function(T_pred)

        return T_pred, br

    def predict_proba(self, X, variational_inference: bool = False):
        X = (X - self.X_mean) / self.X_std
        dataset = create_torch_dataset(X, window_size=self.window_size)
        data_loader = DataLoader(dataset, batch_size=len(X), shuffle=False)

        self.eval()

        with torch.no_grad():
            for (X_batch,) in data_loader:
                br_batch = self.br_unit(X_batch, variational_inference)
                outputs, _ = self(br_batch)

        return outputs.cpu().numpy()

    def predict(self, X, variational_inference: bool = False):
        return np.round(self.predict_proba(X, variational_inference))


class IncreaseAlphaCallback:
    """
    A callback to adjust alpha over time as done in Bica et al. (2020)"""

    def __init__(
        self,
        treatment_head: TreatmentHead,
        alpha_max: float,
        max_epochs: int,
        rate: str = "exp",
    ) -> None:
        self.treatment_head = treatment_head
        self.alpha_max = alpha_max
        self.rate = rate
        self.max_epochs = max_epochs

    def __call__(self, epoch: int) -> None:
        p = float(epoch + 1) / float(self.max_epochs)
        if self.rate == "lin":
            self.treatment_head.alpha = p * self.alpha_max
        elif self.rate == "exp":
            self.treatment_head.alpha = (
                2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            ) * self.alpha_max


class SaveBRCallback:
    """
    A callback to save balanced representation at specified epochs
    """

    def __init__(
        self,
        br_unit: BRUnit,
        save_every: int = 10,
    ):
        self.save_every = save_every
        self.br_unit = br_unit
        self.br_dict = {}

    def __call__(
        self,
        epoch: int,
        X: pd.DataFrame,
        T: Union[pd.Series, pd.DataFrame],
        window_size: int,
        set: str = "train",
    ):
        if epoch % self.save_every != 0:
            return
        else:
            dataset = create_torch_dataset(X, window_size=window_size)
            data_loader = DataLoader(dataset, batch_size=len(X), shuffle=False)
            with torch.no_grad():
                for (X_batch,) in data_loader:
                    br = self.br_unit(X_batch)
                    self.br_dict[(epoch, set)] = (
                        pd.DataFrame(br.cpu().numpy(), index=X.index),
                        T,
                    )


class TwoHeadedEncoder(nn.Module):
    """
    Encoder with one treatment and one outcome head.
    """

    def __init__(
        self,
        input_size: int,
        seq_hidden_size: int,
        br_size: int,
        fc_hidden_size: int,
        dim_treatments: int,
        output_size: int,
        seq_num_layers: int,
        input_dropout: float = 0,
        dropout: float = 0.1,
        alpha: float = 1.0,
        balancing: Optional[str] = "grad_reverse",
        window_size: int = 7,
        activation_function_outcome: str = "linear",
        activation_function_treatment: str = "sigmoid",
    ) -> None:
        super(TwoHeadedEncoder, self).__init__()
        self.seq_hidden_size = seq_hidden_size
        self.fc_hidden_size = fc_hidden_size
        self.seq_num_layers = seq_num_layers
        self.input_dropout = input_dropout

        self.br_unit = BRUnit(
            input_size=input_size,
            seq_hidden_size=seq_hidden_size,
            br_size=br_size,
            seq_num_layers=seq_num_layers,
            dropout=dropout,
            input_dropout=input_dropout,
        )
        self.outcome_head = OutcomeHead(
            br_unit=self.br_unit,
            fc_hidden_size=fc_hidden_size,
            dim_treatments=dim_treatments,
            output_size=output_size,
            activation_function=activation_function_outcome,
            window_size=window_size,
        )
        self.treatment_head = TreatmentHead(
            br_unit=self.br_unit,
            fc_hidden_size=fc_hidden_size,
            dim_treatments=dim_treatments,
            balancing=balancing,
            alpha=alpha,
            activation_function=activation_function_treatment,
            window_size=window_size,
        )

    def forward(
        self, X: torch.Tensor, T: torch.Tensor, variational_inference: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        br = self.br_unit(X, variational_inference)
        T_pred, _ = self.treatment_head(br)
        Y_pred, br = self.outcome_head(br, T)
        return T_pred, Y_pred, br


class ThreeHeadedEncoder(nn.Module):
    """
    Encoder with one treatment and two outcome heads.
    """

    def __init__(
        self,
        input_size: int,
        seq_hidden_size: int,
        br_size: int,
        fc_hidden_size: int,
        dim_treatments: int,
        output_size: int,
        seq_num_layers: int,
        input_dropout: float = 0,
        dropout: float = 0.1,
        alpha: float = 1.0,
        balancing: Optional[str] = "grad_reverse",
        window_size: int = 7,
        activation_function_outcome: str = "linear",
        activation_function_treatment: str = "sigmoid",
    ) -> None:
        super(ThreeHeadedEncoder, self).__init__()
        self.seq_hidden_size = seq_hidden_size
        self.fc_hidden_size = fc_hidden_size
        self.seq_num_layers = seq_num_layers
        self.input_dropout = input_dropout

        self.br_unit = BRUnit(
            input_size=input_size,
            seq_hidden_size=seq_hidden_size,
            br_size=br_size,
            seq_num_layers=seq_num_layers,
            dropout=dropout,
            input_dropout=input_dropout,
        )
        self.outcome_head1 = OutcomeHead(
            br_unit=self.br_unit,
            fc_hidden_size=fc_hidden_size,
            dim_treatments=0,
            output_size=output_size,
            activation_function=activation_function_outcome,
            window_size=window_size,
        )
        self.outcome_head0 = OutcomeHead(
            br_unit=self.br_unit,
            fc_hidden_size=fc_hidden_size,
            dim_treatments=0,
            output_size=output_size,
            activation_function=activation_function_outcome,
            window_size=window_size,
        )
        self.treatment_head = TreatmentHead(
            br_unit=self.br_unit,
            fc_hidden_size=fc_hidden_size,
            dim_treatments=dim_treatments,
            balancing=balancing,
            alpha=alpha,
            activation_function=activation_function_treatment,
            window_size=window_size,
        )

    def forward(
        self, X: torch.Tensor, T: torch.Tensor, variational_inference: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        br = self.br_unit(X, variational_inference)
        T_pred, _ = self.treatment_head(br)

        # divide br based on T
        br0, br1 = br[T[:, 0] == 0, :], br[T[:, 0] == 1, :]
        Y_pred0, _ = self.outcome_head0(br0, None)
        Y_pred1, br = self.outcome_head1(br1, None)
        return T_pred, (Y_pred0, Y_pred1), br


class RepresentationEncoder(BaseEstimator):
    """
    A class that encapsulates the training of a representation encoder.
    """

    def __init__(
        self,
        input_size: int,
        dim_treatments: int,
        output_size: int,
        seq_num_layers: int = 1,
        seq_hidden_size: int = 64,
        br_size: int = 64,
        fc_hidden_size: int = 64,
        balancing: Optional[str] = "grad_reverse",
        input_dropout: float = 0,
        dropout: float = 0.1,
        initial_alpha: float = 0.0,
        alpha_max: float = 1.0,
        adapt_alpha: bool = True,
        save_br_every: Optional[int] = 10,
        encoder_type="two-headed",
        learning_rate=0.001,
        num_epochs: int = 250,
        batch_size: int = 32,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        window_size: int = 7,
        shuffle: bool = False,
        patience: int = 50,
        verbose: bool = False,
        loss_outcome: str = "MSE",
        loss_treatment: str = "BCE",
        **kwargs,
    ):
        if encoder_type == "two-headed":
            self.model = TwoHeadedEncoder(
                input_size=input_size,
                seq_hidden_size=seq_hidden_size,
                br_size=br_size,
                fc_hidden_size=fc_hidden_size,
                dim_treatments=dim_treatments,
                output_size=output_size,
                seq_num_layers=seq_num_layers,
                input_dropout=input_dropout,
                dropout=dropout,
                alpha=initial_alpha,
                balancing=balancing,
            )
        elif encoder_type == "three-headed":
            self.model = ThreeHeadedEncoder(
                input_size=input_size,
                seq_hidden_size=seq_hidden_size,
                br_size=br_size,
                fc_hidden_size=fc_hidden_size,
                dim_treatments=dim_treatments,
                output_size=output_size,
                seq_num_layers=seq_num_layers,
                input_dropout=input_dropout,
                dropout=dropout,
                alpha=initial_alpha,
                balancing=balancing,
            )
        else:
            raise ValueError(
                "Only 'two-headed' and 'three-headed' are valid as encoder_type at the moment."
            )
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.window_size = window_size
        self.shuffle = shuffle
        self.patience = patience
        self.verbose = verbose
        self.output_size = output_size
        self.dim_treatments = dim_treatments
        self.adapt_alpha = adapt_alpha

        # Loss and optimizer
        if loss_outcome == "MSE":
            self.outcome_criterion = nn.MSELoss()
        else:
            raise ValueError(
                "Loss functions other than 'MSE' not supported for outcome at the moment."
            )
        if loss_treatment == "BCE":
            self.treatment_criterion = nn.BCELoss()
        else:
            raise ValueError(
                "Loss functions other than 'BCE' not supported for treatment at the moment."
            )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        if self.adapt_alpha:
            # callback to increase alpha over time
            self.increase_alpha_callback = IncreaseAlphaCallback(
                self.model.treatment_head, alpha_max=alpha_max, max_epochs=num_epochs
            )
        if save_br_every is not None:
            # callback to save balanced representation during training
            self.save_br_callback = SaveBRCallback(
                br_unit=self.model.br_unit,
                save_every=save_br_every,
            )
        else:
            self.save_br_callback = None

    def fit(
        self,
        X: pd.DataFrame,
        T: Union[pd.Series, pd.DataFrame],
        Y: pd.DataFrame,
        X_val: pd.DataFrame = pd.DataFrame(),
        T_val: Union[pd.Series, pd.DataFrame] = pd.DataFrame(),
        Y_val: pd.DataFrame = pd.DataFrame(),
        sample_weight: Optional[pd.Series] = None,
    ):
        # scale X and y
        self.X_mean = self.model.treatment_head.X_mean = X.mean()
        self.X_std = self.model.treatment_head.X_std = X.std() + 0.00000001
        self.Y_mean = Y.mean()
        self.Y_std = Y.std() + 0.00000001
        X = (X - self.X_mean) / self.X_std
        Y = (Y - self.Y_mean) / self.Y_std

        if not X_val.empty:
            # scale X_val and Y_val
            X_val = (X_val - self.X_mean) / self.X_std
            Y_val = (Y_val - self.Y_mean) / self.Y_std

        self.best_loss = float("inf")
        self.best_epoch = self.num_epochs
        self.loss_log = {
            "train": [],
            "train_outcome": [],
            "train_treatment": [],
            "val": [],
            "val_outcome": [],
            "val_treatment": [],
            "alpha": [],
        }
        counter = 0

        for epoch in range(self.num_epochs):
            # Training loop
            self.model.train()
            train_loss, train_loss_outcome, train_loss_treatment = 0.0, 0.0, 0.0
            dataset_train = create_torch_dataset(
                X, self.window_size, Y, T, sample_weight, sort_by_date=True
            )
            data_loader_train = DataLoader(
                dataset_train, batch_size=self.batch_size, shuffle=self.shuffle
            )
            for X_batch, Y_batch, T_batch in data_loader_train:
                self.optimizer.zero_grad()
                # Forward pass
                T_pred, Y_pred, _ = self.model(X_batch, T_batch)
                if type(Y_pred) == tuple:
                    # for two outcome heads
                    loss_outcome = torch.add(
                        torch.nan_to_num(
                            self.outcome_criterion(
                                Y_pred[0], Y_batch[T_batch[:, 0] == 0]
                            )
                        ),
                        torch.nan_to_num(
                            self.outcome_criterion(
                                Y_pred[1], Y_batch[T_batch[:, 0] == 1]
                            )
                        ),
                    )
                else:
                    # for one outcome head
                    loss_outcome = self.outcome_criterion(Y_pred, Y_batch)
                loss_treatment = self.treatment_criterion(
                    T_pred,
                    T_batch,  # sample_weight_batch
                )  # , epoch)
                loss = loss_outcome + loss_treatment
                train_loss_outcome = train_loss_outcome + loss_outcome.item()
                train_loss_treatment = train_loss_treatment + loss_treatment.item()
                train_loss = train_loss + loss.item()

                # Calculate L1 regularization loss
                l1_loss = 0
                for param in self.model.parameters():
                    l1_loss += torch.norm(param, 1)  # L1 norm

                # Calculate L1 regularization loss
                l2_loss = 0
                for param in self.model.parameters():
                    l2_loss += torch.norm(param, 2)  # L2 norm

                # Add L1/L2 regularization to the loss
                loss = loss + self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

            train_loss /= len(data_loader_train)
            train_loss_outcome /= len(data_loader_train)
            train_loss_treatment /= len(data_loader_train)
            self.loss_log["train"].append(train_loss)
            self.loss_log["train_outcome"].append(train_loss_outcome)
            self.loss_log["train_treatment"].append(train_loss_treatment)
            self.loss_log["alpha"].append(self.model.treatment_head.alpha)

            # save balanced representation for training data
            if self.save_br_callback is not None:
                self.save_br_callback(
                    epoch=epoch, X=X, T=T, window_size=self.window_size, set="train"
                )

            val_loss, val_loss_outcome, val_loss_treatment = 0.0, 0.0, 0.0
            if not X_val.empty:
                # Validation loop for early stopping if val data provided
                self.model.eval()
                dataset_val = create_torch_dataset(
                    X_val, self.window_size, Y_val, T_val
                )
                data_loader_val = DataLoader(
                    dataset_val, batch_size=self.batch_size, shuffle=False
                )
                with torch.no_grad():
                    for X_batch, Y_batch, T_batch in data_loader_val:
                        T_pred, Y_pred, _ = self.model(X_batch, T_batch)
                        if type(Y_pred) == tuple:
                            # for two outcome heads
                            loss_outcome = torch.add(
                                torch.nan_to_num(
                                    self.outcome_criterion(
                                        Y_pred[0], Y_batch[T_batch[:, 0] == 0]
                                    )
                                ),
                                torch.nan_to_num(
                                    self.outcome_criterion(
                                        Y_pred[1], Y_batch[T_batch[:, 0] == 1]
                                    )
                                ),
                            )
                        else:
                            # for one outcome head
                            loss_outcome = self.outcome_criterion(Y_pred, Y_batch)
                        loss_treatment = self.treatment_criterion(
                            T_pred, T_batch
                        )  # , epoch)
                        loss = loss_outcome + loss_treatment
                        val_loss_outcome = val_loss_outcome + loss_outcome.item()
                        val_loss_treatment = val_loss_treatment + loss_treatment.item()
                        val_loss = val_loss + loss.item()

                    val_loss /= len(data_loader_val)
                    val_loss_outcome /= len(data_loader_val)
                    val_loss_treatment /= len(data_loader_val)
                    self.loss_log["val"].append(val_loss)
                    self.loss_log["val_outcome"].append(val_loss_outcome)
                    self.loss_log["val_treatment"].append(val_loss_treatment)

                    # save balanced representation for validation data
                    if self.save_br_callback is not None:
                        self.save_br_callback(
                            epoch=epoch,
                            X=X_val,
                            T=T_val,
                            window_size=self.window_size,
                            set="val",
                        )

                    # Check for early stopping
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.best_epoch = epoch
                        self.best_model = copy.deepcopy(self.model)
                        counter = 0
                    else:
                        counter += 1
                        if counter >= self.patience:
                            if self.verbose:
                                print(
                                    "\nEarly stopping triggered at epoch {}. Training stopped.".format(
                                        epoch
                                    )
                                )
                            break
            else:
                self.best_loss = train_loss
                self.best_model = self.model

            if self.adapt_alpha:
                # increase alpha
                self.increase_alpha_callback(epoch=epoch)

            if self.verbose:
                print(
                    f"Epoch {epoch}/{self.num_epochs}: train loss={train_loss:.7f}, val loss={val_loss:.7f}, alpha={self.model.treatment_head.alpha}, early stopping counter={counter}/{self.patience}",
                    end="\r",
                )

        self.is_fitted_ = True
        return self

    def get_y_learner(self):
        if type(self.best_model) == ThreeHeadedEncoder:
            return self.best_model.outcome_head0, self.best_model.outcome_head1
        else:
            return self.best_model.outcome_head

    def get_t_learner(self):
        return self.best_model.treatment_head

    def predict(self, X: pd.DataFrame, T: Optional[Union[pd.Series, pd.DataFrame]]):
        if type(self.best_model) == ThreeHeadedEncoder:
            return [y_learner.predict(X, T) for y_learner in self.get_y_learner()]
        else:
            return self.get_y_learner().predict(X, T)


class DragonnetEncoder(RepresentationEncoder):
    """A recurrent version of the Dragonnet (Shi et al., 2019) with two outcome heads and no balancing in the treatment head."""

    def __init__(self, **kwargs):
        super().__init__(
            balancing=None,
            encoder_type="three-headed",
            adapt_alpha=False,
            initial_alpha=1.0,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        # store mean and std for standardization of test data (also in treatment and outcome heads for predictions)
        (
            self.model.outcome_head0.X_mean,
            self.model.outcome_head0.X_std,
            self.model.outcome_head0.Y_mean,
            self.model.outcome_head0.Y_std,
        ) = (X.mean(), X.std() + 0.00000001, Y.mean(), Y.std() + 0.00000001)
        (
            self.model.outcome_head1.X_mean,
            self.model.outcome_head1.X_std,
            self.model.outcome_head1.Y_mean,
            self.model.outcome_head1.Y_std,
        ) = (X.mean(), X.std() + 0.00000001, Y.mean(), Y.std() + 0.00000001)
        super().fit(X=X, Y=Y, **kwargs)


class CRNEncoder(RepresentationEncoder):
    """The encoder part of the CRN (Bica et al., 2020) with one outcome head and balancing via gradient reversal."""

    def __init__(self, **kwargs):
        super().__init__(
            balancing="grad_reverse",
            encoder_type="two-headed",
            adapt_alpha=True,
            initial_alpha=0.0,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        # store mean and std for standardization of test data (also in treatment and outcome heads for predictions)
        (
            self.model.outcome_head.X_mean,
            self.model.outcome_head.X_std,
            self.model.outcome_head.Y_mean,
            self.model.outcome_head.Y_std,
        ) = (X.mean(), X.std() + 0.00000001, Y.mean(), Y.std() + 0.00000001)

        super().fit(X=X, Y=Y, **kwargs)
