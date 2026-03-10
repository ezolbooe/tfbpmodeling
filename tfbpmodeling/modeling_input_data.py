import logging
import os

import pandas as pd
from patsy import PatsyError, dmatrix
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("main")


class ModelingInputData:
    """
    Container for response and predictor data used in modeling transcription factor
    perturbation experiments.

    This class handles:
        - Validation and synchronization of response and predictor DataFrames
        based on a shared feature identifier.
        - Optional blacklisting of features, including the perturbed
            transcription factor.
        - Optional feature selection based on the top N strongest binding signals
            (as ranked from a specific TF column in the predictor matrix).
        - Application of masking logic to restrict modeling to selected features.

    """

    def __init__(
        self,
        response_df: pd.DataFrame,
        predictors_df: pd.DataFrame,
        perturbed_tf: str,
        feature_col: str = "target_symbol",
        feature_blacklist: list[str] | None = None,
        top_n: int | None = None,
        stage2_set_zero: bool = False, 
    ):
        """
        Initialize ModelingInputData with response and predictor matrices. Note that the
        response and predictor dataframes will be subset down to the features in common
        between them, by index. The rows in both dataframes will also be ordered such
        that they match, again by index.

        :param response_df: A two column DataFrame containing the `feature_col` and
            numeric column representing the response variable.
        :param predictors_df: A Dataframe containing the `feature_col` and predictor
            numeric columns that represent the predictor variables.
        :param perturbed_tf: Name of the perturbed TF. **Note**: this must exist as a
            column in predictors_df.
        :param feature_col: Name of the column to use as the feature index. This column
            must exist in both the response and predictor DataFrames.
            (default: "target_symbol").
        :param feature_blacklist: List of feature names to exclude from analysis.
        :param top_n: If specified, retain only the top N features with the strongest
            binding scores for the perturbed TF. If this is passed on initialization,
            then the top_n_masked is set to True by default. If you wish to extract
            unmasked data, you can set `object.top_n_masked = False`. The mask can be
            toggled on and off at will.

        """
        if not isinstance(response_df, pd.DataFrame):
            raise ValueError("response_df must be a DataFrame.")
        if not isinstance(predictors_df, pd.DataFrame):
            raise ValueError("predictors_df must be a DataFrame.")
        if not isinstance(perturbed_tf, str):
            raise ValueError("perturbed_tf must be a string representing the TF name.")
        if not isinstance(feature_col, str):
            raise ValueError(
                "feature_col must be a string representing the feature name."
            )
        if feature_blacklist is not None and not isinstance(feature_blacklist, list):
            raise ValueError("feature_blacklist must be a list or None.")
        if top_n is not None and not isinstance(top_n, int):
            raise ValueError("top_n must be an integer or None.")

        self.perturbed_tf = perturbed_tf
        self.feature_col = feature_col
        self._top_n_masked = False

        # Ensure feature_blacklist is a list
        if feature_blacklist is None:
            feature_blacklist = []

        # Ensure perturbed_tf is in the blacklist
        if perturbed_tf not in feature_blacklist:
            logger.warning(
                f"Perturbed TF '{perturbed_tf}' not in blacklist. "
                f"Adding to blacklist. Setting blacklist_masked to True. "
                f"If you do not wish to blacklist the perturbed TF, "
                f"set blacklist_masked to False."
            )
            feature_blacklist.append(perturbed_tf)

        self.feature_blacklist = set(feature_blacklist)
        self.blacklist_masked = bool(self.feature_blacklist)

        # Ensure the response and predictors only contain common features
        self.response_df = response_df
        self.predictors_df = predictors_df
        self.stage2_set_zero = stage2_set_zero
        # Assign top_n value
        self.top_n = top_n

    @property
    def response_df(self) -> pd.DataFrame:
        response_df = self._response_df.copy()

        # Always drop blacklisted features
        if self.blacklist_masked:
            response_df = response_df.loc[
                # Use intersection to ensure we only try to drop what exists
                response_df.index.difference(self.feature_blacklist)
            ]

        # Handle Top-N logic
        if self.top_n_masked and self.top_n_features:
            if self.stage2_set_zero:
                # DO NOTHING to the values. 
                # Because we didn't use .loc[self.top_n_features], 
                # all rows remain with their original LRR values.
                pass
            else:
                # Original behavior: Drop the rows to match the filtered predictors
                response_df = response_df.loc[self.top_n_features]
        
        # Always ensure the response order matches the predictors order
        response_df = response_df.reindex(self.predictors_df.index)

        return response_df

    @response_df.setter
    def response_df(self, value: pd.DataFrame) -> None:
        """
        Set the response DataFrame and enforce schema and consistency constraints.

        The input DataFrame must contain:
        - The feature identifier column specified by `feature_col`
        - Exactly one numeric column (excluding `feature_col`)

        After setting, the internal response matrix will use `feature_col` as its index.
        If the predictors DataFrame has already been set, both matrices will be subset
        and reordered to retain only their shared features.

        :param value: DataFrame containing response values.
        :raises ValueError: If input is not a DataFrame or does not contain
            exactly one numeric column.
        :raises KeyError: If required columns are missing.

        """
        if not isinstance(value, pd.DataFrame):
            raise ValueError("response_df must be a DataFrame.")
        if self.feature_col not in value.columns:
            raise KeyError(
                f"Feature column '{self.feature_col}' not found in response DataFrame."
            )

        # Ensure the response DataFrame has exactly one numeric
        # column (excluding feature_col)
        numeric_cols = value.drop(columns=[self.feature_col]).select_dtypes(
            include="number"
        )
        if numeric_cols.shape[1] != 1:
            raise ValueError(
                "Response DataFrame must have exactly one numeric "
                "column other than the feature_col."
            )

        logger.info(f"Response column names: {numeric_cols.columns}")

        self._response_df = value.set_index(self.feature_col)
        if hasattr(self, "_predictors_df") and self._predictors_df is not None:
            self._set_common_features_and_order()

    @property
    def predictors_df(self) -> pd.DataFrame:
        predictors_df = self._predictors_df.copy()

        # Apply blacklist masking (usually still want to drop these)
        if self.blacklist_masked:
            predictors_df = predictors_df.loc[
                predictors_df.index.difference(self.feature_blacklist)
            ]

        # New Top-N Logic
        if self.top_n_masked and self.top_n_features:
            if self.stage2_set_zero:
                # Set all rows NOT in top_n_features to 0.0
                mask = ~predictors_df.index.isin(self.top_n_features)
                predictors_df.loc[mask, :] = 0.0 
            else:
                # Original behavior: Drop the rows entirely
                predictors_df = predictors_df.loc[self.top_n_features, :]

        return predictors_df

    @predictors_df.setter
    def predictors_df(self, value: pd.DataFrame) -> None:
        """
        Set the predictors DataFrame and enforce schema constraints.

        The input DataFrame must include the `feature_col` (used as the index)
        and the column corresponding to the perturbed transcription factor.
        After setting, the response and predictor matrices will be aligned to
        retain only common features.

        :param value: DataFrame containing predictor features.
        :raises ValueError: If input is not a DataFrame.
        :raises KeyError: If required columns are missing.

        """
        if not isinstance(value, pd.DataFrame):
            raise ValueError("predictors_df must be a DataFrame.")
        if self.feature_col not in value.columns:
            raise KeyError(
                f"Feature column '{self.feature_col}' "
                "not found in predictors DataFrame."
            )
        if self.perturbed_tf not in value.columns:
            raise KeyError(
                f"Perturbed TF '{self.perturbed_tf}' not found in predictor index."
            )

        self._predictors_df = value.set_index(self.feature_col)
        if hasattr(self, "_response_df") and self._response_df is not None:
            self._set_common_features_and_order()

    @property
    def top_n(self) -> int | None:
        """
        Get the threshold for top-ranked feature selection.

        If set to an integer, this defines how many of the highest-ranked features
        (based on `predictors_df[self.perturbed_tf]`) should be retained. Ranking is
        descending (higher values rank higher). If the cutoff falls on a tie,
        fewer than N features may be selected to preserve a consistent threshold. The
        most impactful tie is when the majority of the lower ranked features have
        the same value, eg an enrichment of 0 or pvalue of 1.0.

        If set to None, top-N feature selection is disabled.

        Note: Whether top-N filtering is actively applied depends on the
        `top_n_masked` attribute. You can set `top_n_masked = False` to access the
        unfiltered data, even if `top_n` is set.

        :return: The current top-N threshold or None.

        """
        return self._top_n

    @top_n.setter
    def top_n(self, value: int | None) -> None:
        """
        Set the top-N threshold and update the feature mask.

        :param value: Positive integer or None.
        :raises ValueError: If value is not a positive integer or None.

        """
        # validate that top_n is an int greater than 0
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError("top_n must be a positive integer or None.")
        # if top_n is None, set _top_n to None and _top_n_masked to False
        if value is None:
            self._top_n = value
            self._top_n_masked = False
        # else, find the top_n features according to predictors_df[perturbed_tf]
        else:
            perturbed_df = self.predictors_df.loc[:, self.perturbed_tf]

            # Rank in descending order (higher values get lower ranks)
            ranks = pd.Series(
                rankdata(-perturbed_df, method="average"),
                index=perturbed_df.index,
            )

            # Count occurrences of each unique rank
            rank_counts = ranks.value_counts().sort_index(ascending=True)
            cumulative_counts = rank_counts.cumsum()

            # Find the highest rank where cumulative count ≤ top_n
            selected_rank = (
                rank_counts.index[cumulative_counts <= value].max()
                if (cumulative_counts <= value).any()
                else None
            )

            # Subset based on the determined rank threshold
            selected_features = (
                self.predictors_df.loc[ranks <= selected_rank].index.tolist()
                if selected_rank is not None
                else []
            )

            # Store results and log info
            self._top_n = value
            self.top_n_features = selected_features
            logger.info(
                f"Selected {len(selected_features)} top features based on "
                f"descending ranking of predictors_df['{self.perturbed_tf}']."
            )
            self.top_n_masked = True

    @property
    def top_n_masked(self) -> bool:
        """
        Get the status of top-n feature masking.

        If this is `True`, then
        the top-n feature selection is applied to the predictors and response

        """
        return self._top_n_masked

    @top_n_masked.setter
    def top_n_masked(self, value: bool) -> None:
        """Set the status of top-n feature masking."""
        if not isinstance(value, bool):
            raise ValueError("top_n_masked must be a boolean.")
        if value:
            logger.info("Top-n feature masking enabled.")
        else:
            logger.info("Top-n feature masking disabled.")
        self._top_n_masked = value

    def _set_common_features_and_order(self) -> None:
        """Ensures that the response and predictor dataframes contain only the common
        features and are ordered identically based on `feature_col`."""
        # Identify common features between response_df and predictors_df
        common_feature_set = set(self._response_df.index).intersection(
            set(self._predictors_df.index)
        )

        if not common_feature_set:
            raise ValueError(
                "No common features found between response and predictors DataFrames."
            )

        logger.info(
            f"Common features between response and predictors: "
            f"{len(common_feature_set)}. "
            f"Subsetting and reordering both dataframes."
        )

        # Apply blacklist before subsetting
        if self.blacklist_masked:
            # log the intersect between the common features and the blacklist as the
            # number of blacklisted genes
            logger.info(
                f"Number of blacklisted features: "
                f"{len(common_feature_set.intersection(self.feature_blacklist))}"
            )
            common_feature_set -= self.feature_blacklist

        # Subset both dataframes based on the common features
        response_df_filtered = self._response_df.loc[list(common_feature_set)]
        predictors_df_filtered = self._predictors_df.loc[list(common_feature_set)]

        # Ensure response_df is ordered according to predictors_df
        response_df_ordered = response_df_filtered.loc[predictors_df_filtered.index]

        # raise an error if the indices of the response and predictors
        # do not match after filtering
        if not response_df_ordered.index.equals(predictors_df_filtered.index):
            raise ValueError(
                "Indices of response_df and predictors_df do not match after "
                "filtering for common features. Please check your input data."
            )

        # Set the response and predictor DataFrames with ordered features
        self._response_df = response_df_ordered
        self._predictors_df = predictors_df_filtered

    def get_modeling_data(
        self,
        formula: str,
        add_row_max: bool = False,
        drop_intercept: bool = False,
        scale_by_std: bool = False,
    ) -> pd.DataFrame:
        """
        Get the predictors for modeling, optionally adding a row-wise max feature.

        :param formula: The formula to use for modeling.
        :param add_row_max: Whether to add a row-wise max feature to the predictors.
        :param drop_intercept: If `drop_intercept` is True, "-1" will be appended to
            the formula string. This will drop the intercept (constant) term from
            the model matrix output by patsy.dmatrix. Default is `False`.
        :param scale_by_std: If True, scale the design matrix by standard deviation
            using StandardScaler(with_mean=False, with_std=True). The data is NOT
            centered, so the estimator should still fit an intercept
            (fit_intercept=True).
        :return: The design matrix for modeling. self.response_df can be used for the
            response variable.

        :raises ValueError: If the formula is not provided
        :raises PatsyError: If there is an error in creating the model matrix

        """
        if not formula:
            raise ValueError("Formula must be provided for modeling.")

        if drop_intercept:
            logger.info("Dropping intercept from the patsy model matrix")
            formula += " - 1"

        predictors_df = self.predictors_df  # Apply top-n feature mask

        # Add row-wise max feature if requested
        if add_row_max:
            predictors_df["row_max"] = predictors_df.max(axis=1)

        # Create a design matrix using patsy
        try:
            design_matrix = dmatrix(
                formula,
                data=predictors_df,
                return_type="dataframe",
                NA_action="raise",
            )
        except PatsyError as exc:
            logger.error(
                f"Error in creating model matrix with formula '{formula}': {exc}"
            )
            raise

        if scale_by_std:
            logger.info("Center matrix = `False`. Scale matrix = `True`")
            scaler = StandardScaler(with_mean=False, with_std=True)
            scaled_values = scaler.fit_transform(design_matrix)
            design_matrix = pd.DataFrame(
                scaled_values, index=design_matrix.index, columns=design_matrix.columns
            )

        logger.info(f"Design matrix columns: {list(design_matrix.columns)}")

        return design_matrix

    @classmethod
    def from_files(
        cls,
        response_path: str,
        predictors_path: str,
        perturbed_tf: str,
        feature_col: str = "target_symbol",
        feature_blacklist_path: str | None = None,
        top_n: int = 600,
        stage2_set_zero: bool = False,
    ) -> "ModelingInputData":
        """
        Load response and predictor data from files. This would be considered an
        overloaded constructor in other languages. The input files must be able to be
        read into objects that satisfy the __init__ method -- see __init__ docs.

        :param response_path: Path to the response file (CSV).
        :param predictors_path: Path to the predictors file (CSV).
        :param perturbed_tf: The perturbed TF.
        :param feature_col: The column name representing features.
        :param feature_blacklist_path: Path to a file containing a list of features to
            exclude.
        :param top_n: Maximum number of features for top-n selection.
        :return: An instance of ModelingInputData.
        :raises FileNotFoundError: If the response or predictor files are missing.

        """
        if not os.path.exists(response_path):
            raise FileNotFoundError(f"Response file '{response_path}' does not exist.")
        if not os.path.exists(predictors_path):
            raise FileNotFoundError(
                f"Predictors file '{predictors_path}' does not exist."
            )

        response_df = pd.read_csv(response_path)
        predictors_df = pd.read_csv(predictors_path)

        # Load feature blacklist if provided
        feature_blacklist: list[str] = []
        if feature_blacklist_path:
            if not os.path.exists(feature_blacklist_path):
                raise FileNotFoundError(
                    f"Feature blacklist file '{feature_blacklist_path}' does not exist."
                )
            with open(feature_blacklist_path) as f:
                feature_blacklist = [line.strip() for line in f if line.strip()]

        return cls(
            response_df,
            predictors_df,
            perturbed_tf,
            feature_col,
            feature_blacklist,
            top_n,
            stage2_set_zero,
        )
