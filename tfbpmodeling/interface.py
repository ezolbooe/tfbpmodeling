import argparse
import json
import logging
import os

import joblib
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold

from tfbpmodeling.bootstrap_stratified_cv import bootstrap_stratified_cv_modeling
from tfbpmodeling.bootstrap_stratified_cv_loop import bootstrap_stratified_cv_loop
from tfbpmodeling.bootstrapped_input_data import BootstrappedModelingInputData
from tfbpmodeling.evaluate_interactor_significance_lassocv import (
    evaluate_interactor_significance_lassocv,
)
from tfbpmodeling.evaluate_interactor_significance_linear import (
    evaluate_interactor_significance_linear,
)
from tfbpmodeling.modeling_input_data import ModelingInputData
from tfbpmodeling.stratification_classification import stratification_classification
from tfbpmodeling.stratified_cv import stratified_cv_modeling
from tfbpmodeling.utils.exclude_predictor_variables import exclude_predictor_variables

logger = logging.getLogger("main")


class CustomHelpFormatter(argparse.HelpFormatter):
    """
    This could be used to customize the help message formatting for the argparse parser.

    Left as a placeholder.

    """


def linear_perturbation_binding_modeling(args):
    """
    :param args: Command-line arguments containing input file paths and parameters.
    """
    if not isinstance(args.max_iter, int) or args.max_iter < 1:
        raise ValueError("The `max_iter` parameter must be a positive integer.")

    max_iter = int(args.max_iter)

    logger.info(f"estimator max_iter: {max_iter}.")

    logger.info("Step 1: Preprocessing")

    # validate input files/dirs
    if not os.path.exists(args.response_file):
        raise FileNotFoundError(f"File {args.response_file} does not exist.")
    if not os.path.exists(args.predictors_file):
        raise FileNotFoundError(f"File {args.predictors_file} does not exist.")
    if os.path.exists(args.output_dir):
        logger.warning(f"Output directory {args.output_dir} already exists.")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory created at {args.output_dir}")

    # the output subdir is where the output of this modeling run will be saved
    output_subdir = os.path.join(
        args.output_dir, os.path.join(args.perturbed_tf + args.output_suffix)
    )
    if os.path.exists(output_subdir):
        raise FileExistsError(
            f"Directory {output_subdir} already exists. "
            "Please specify a different `output_dir`."
        )
    else:
        os.makedirs(output_subdir, exist_ok=True)
        logger.info(f"Output subdirectory created at {output_subdir}")

    # instantiate a estimator
    estimator = LassoCV(
        fit_intercept=True,
        selection="random",
        n_alphas=100,
        random_state=42,
        n_jobs=args.n_cpus,
        max_iter=max_iter,
    )

    input_data = ModelingInputData.from_files(
        response_path=args.response_file,
        predictors_path=args.predictors_file,
        perturbed_tf=args.perturbed_tf,
        feature_blacklist_path=args.blacklist_file,
        top_n=args.top_n,
        stage2_set_zero=args.stage2_set_zero,
    )

    logger.info("Step 2: Bootstrap LassoCV on all data, full interactor model")

    # Unset the top n masking -- we want to use all the data for the first round
    # modeling
    input_data.top_n_masked = False

    # extract a list of predictor variables, which are the columns of the predictors_df
    predictor_variables = input_data.predictors_df.columns.drop(input_data.perturbed_tf)

    # drop any variables which are in args.exclude_interactor_variables
    predictor_variables = exclude_predictor_variables(
        list(predictor_variables), args.exclude_interactor_variables
    )

    # create a list of interactor terms with the perturbed_tf as the first term
    interaction_terms = [
        f"{input_data.perturbed_tf}:{var}" for var in predictor_variables
    ]

    # Construct the full interaction formula, ie perturbed_tf + perturbed_tf:other_tf1 +
    # perturbed_tf:other_tf2 + ... . perturbed_tf main effect only added if
    # --ptf_main_effect is passed.
    if args.ptf_main_effect:
        logger.info("adding pTF main effect to `all_data_formula`")
        all_data_formula = (
            f"{input_data.perturbed_tf} + {' + '.join(interaction_terms)}"
        )
    else:
        all_data_formula = " + ".join(interaction_terms)


    
    if args.squared_pTF:
        # if --squared_pTF is passed, then add the squared perturbed TF to the formula
        squared_term = f"I({input_data.perturbed_tf} ** 2)"
        logger.info(f"Adding squared term to model formula: {squared_term}")
        all_data_formula += f" + {squared_term}"

    if args.cubic_pTF:
        # if --cubic_pTF is passed, then add the cubic perturbed TF to the formula
        cubic_term = f"I({input_data.perturbed_tf} ** 3)"
        logger.info(f"Add cubic term to model formula: {cubic_term}")
        all_data_formula += f" + {cubic_term}"


    # if --row_max is passed, then add "row_max" to the formula
    if args.row_max:
        logger.info("Adding `row_max` to the all data model formula")
        all_data_formula += " + row_max"

    # if --add_model_variables is passed, then add the variables to the formula
    if args.add_model_variables:
        logger.info(
            f"Adding model variables to the all data model "
            f"formula: {args.add_model_variables}"
        )
        all_data_formula += " + " + " + ".join(args.add_model_variables)

    logger.debug(f"All data formula: {all_data_formula}")

    # create the bootstrapped data.
    bootstrapped_data_all = BootstrappedModelingInputData(
        response_df=input_data.response_df,
        model_df=input_data.get_modeling_data(
            all_data_formula,
            add_row_max=args.row_max,
            drop_intercept=True,
            scale_by_std=args.scale_by_std,
        ),
        n_bootstraps=args.n_bootstraps,
        normalize_sample_weights=args.normalize_sample_weights,
        random_state=args.random_state,
    )

    logger.info(
        f"Running bootstrap LassoCV on all data with {args.n_bootstraps} bootstraps"
    )

    if args.skip_1st_stage:
        logger.info("Skipping Stage 1 bootstrap filtering. Using all terms for Stage 2.")
        # Simply use the full formula we built for Step 2
        all_data_sig_coefs_formula = all_data_formula
    else:
        if args.iterative_dropout:
            logger.info("Using iterative dropout modeling for all data results.")
            all_data_results = bootstrap_stratified_cv_loop(
                bootstrapped_data=bootstrapped_data_all,
                perturbed_tf_series=input_data.predictors_df[input_data.perturbed_tf],
                estimator=estimator,
                ci_percentile=float(args.all_data_ci_level),
                stabilization_ci_start=args.stabilization_ci_start,
                bins=args.bins,
                output_dir=output_subdir,
            )
        else:
            logger.info("Using standard bootstrap modeling for all data results.")
            all_data_results = bootstrap_stratified_cv_modeling(
                bootstrapped_data=bootstrapped_data_all,
                perturbed_tf_series=input_data.predictors_df[input_data.perturbed_tf],
                estimator=estimator,
                ci_percentiles=[float(args.all_data_ci_level)],
                bins=args.bins,
            )
        # create the all data object output subdir
        all_data_output = os.path.join(output_subdir, "all_data_result_object")
        os.makedirs(all_data_output, exist_ok=True)

        logger.info(f"Serializing all data results to {all_data_output}")
        all_data_results.serialize("result_obj", all_data_output)

        # Extract the coefficients that are significant at the specified confidence level
        all_data_sig_coefs = all_data_results.extract_significant_coefficients(
            ci_level=args.all_data_ci_level,
        )

        logger.info(f"all_data_sig_coefs: {all_data_sig_coefs}")

        if not all_data_sig_coefs:
            logger.warning(
                f"No significant coefficients found at {args.all_data_ci_level}% "
                "confidence level. Exiting."
            )
            return

        # write all_data_sig_coefs to a json file
        all_data_ci_str = str(args.all_data_ci_level).replace(".", "-")
        all_data_output_file = os.path.join(
            output_subdir, f"all_data_significant_{all_data_ci_str}.json"
        )
        logger.info(f"Writing the all data significant results to {all_data_output_file}")
        with open(
            all_data_output_file,
            "w",
        ) as f:
            json.dump(all_data_sig_coefs, f, indent=4)

        # extract the significant coefficients and create a formula.
        all_data_sig_coefs_formula = f"{' + '.join(all_data_sig_coefs.keys())}"
        logger.debug(f"`all_data_sig_coefs_formula` formula: {all_data_sig_coefs_formula}")

        logger.info(
            "Step 3: Bootstrap LassoCV on the significant coefficients "
            "from the all data model. This produces the best model for all data"
        )

        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        classes = stratification_classification(
            input_data.predictors_df[input_data.perturbed_tf].squeeze(),
            bins=args.bins,
        )

        best_all_data_model_df = input_data.get_modeling_data(
            all_data_sig_coefs_formula,
            add_row_max=args.row_max,
            drop_intercept=True,
            scale_by_std=args.scale_by_std,
        )
        best_all_data_model = stratified_cv_modeling(
            input_data.response_df,
            best_all_data_model_df,
            classes=classes,
            estimator=estimator,
            skf=skf,
            sample_weight=None,
        )

        # save the best all data model to file with metadata
        best_model_file = os.path.join(output_subdir, "best_all_data_model.pkl")
        logger.info(f"Saving the best all data model to {best_model_file}")

        # Bundle model with metadata so feature names are preserved
        model_bundle = {
            "model": best_all_data_model,
            "feature_names": list(best_all_data_model_df.columns),
            "formula": all_data_sig_coefs_formula,
            "perturbed_tf": input_data.perturbed_tf,
            "scale_by_std": args.scale_by_std,
            "drop_intercept": True,
        }
        joblib.dump(model_bundle, best_model_file)

    logger.info(
        "Step 4: Running LassoCV on topn data with significant coefficients "
        "from the all data model"
    )

    # apply the top_n masking
    input_data.top_n_masked = True

    # Create the bootstrapped data for the topn modeling
    bootstrapped_data_top_n = BootstrappedModelingInputData(
        response_df=input_data.response_df,
        model_df=input_data.get_modeling_data(
            all_data_sig_coefs_formula,
            add_row_max=args.row_max,
            drop_intercept=True,
            scale_by_std=args.scale_by_std,
        ),
        n_bootstraps=args.n_bootstraps,
        normalize_sample_weights=args.normalize_sample_weights,
        random_state=(
            args.random_state + 10 if args.random_state else args.random_state
        ),
    )

    logger.debug(
        f"Running bootstrap LassoCV on topn data with {args.n_bootstraps} bootstraps"
    )
    topn_results = bootstrap_stratified_cv_modeling(
        bootstrapped_data_top_n,
        input_data.predictors_df[input_data.perturbed_tf],
        estimator=estimator,
        ci_percentiles=[float(args.topn_ci_level)],
    )

    # create the topn data object output subdir
    topn_output = os.path.join(output_subdir, "topn_result_object")
    os.makedirs(topn_output, exist_ok=True)

    logger.info(f"Serializing topn results to {topn_output}")
    topn_results.serialize("result_obj", topn_output)

    # extract the topn_results at the specified confidence level
    topn_output_res = topn_results.extract_significant_coefficients(
        ci_level=args.topn_ci_level
    )

    logger.info(f"topn_output_res: {topn_output_res}")

    if not topn_output_res:
        logger.warning(
            f"No significant coefficients found at {args.topn_ci_level}% "
            "confidence level. Exiting."
        )
        return

    # write topn_output_res to a json file
    topn_ci_str = str(args.topn_ci_level).replace(".", "-")
    topn_output_file = os.path.join(
        output_subdir, f"topn_significant_{topn_ci_str}.json"
    )
    logger.info(f"Writing the topn significant results to {topn_output_file}")
    with open(topn_output_file, "w") as f:
        json.dump(topn_output_res, f, indent=4)

    # ==========================================
    # NEW CODE: Stage 3 (2b) Implementation
    # ==========================================
    if args.stage3_2b:
        logger.info(
            "Step 3 2b: Re-running Stage 1 bootstrap with surviving "
            "interactors and their independent main effects."
        )
        
        # 1. Revert to all data (same configuration as Stage 1)
        input_data.top_n_masked = False
        
        # 2. Extract surviving interactors and their independent variables
        surviving_interactors = list(topn_output_res.keys())
        independent_effects = []
        
        for term in surviving_interactors:
            if ":" in term:
                parts = term.split(":")
                # The independent effect is the part that is NOT the perturbed_tf
                indep = parts[1] if parts[0] == input_data.perturbed_tf else parts[0]
                if indep not in independent_effects:
                    independent_effects.append(indep)
        
        # 3. Build the new Stage 3 formula
        stage3_terms = surviving_interactors + independent_effects
        stage3_formula = " + ".join(stage3_terms)
        
        # Add the baseline/global modifiers back in if they were requested
        if args.ptf_main_effect and input_data.perturbed_tf not in stage3_formula:
            stage3_formula = f"{input_data.perturbed_tf} + {stage3_formula}"
        if args.squared_pTF:
            stage3_formula += f" + I({input_data.perturbed_tf} ** 2)"
        if args.cubic_pTF:
            stage3_formula += f" + I({input_data.perturbed_tf} ** 3)"
        if args.row_max:
            stage3_formula += " + row_max"
        if args.add_model_variables:
            stage3_formula += " + " + " + ".join(args.add_model_variables)

        logger.debug(f"Stage 3 2b formula: {stage3_formula}")

        # 4. Generate bootstrapped data for Stage 3
        bootstrapped_data_stage3 = BootstrappedModelingInputData(
            response_df=input_data.response_df,
            model_df=input_data.get_modeling_data(
                stage3_formula,
                add_row_max=args.row_max,
                drop_intercept=True,
                scale_by_std=args.scale_by_std,
            ),
            n_bootstraps=args.n_bootstraps,
            normalize_sample_weights=args.normalize_sample_weights,
            random_state=args.random_state,
        )

        logger.info(f"Running Stage 3 2b bootstrap LassoCV with {args.n_bootstraps} bootstraps")

        # 5. Run the model using the Stage 1 configurations
        if args.iterative_dropout:
            stage3_results = bootstrap_stratified_cv_loop(
                bootstrapped_data=bootstrapped_data_stage3,
                perturbed_tf_series=input_data.predictors_df[input_data.perturbed_tf],
                estimator=estimator,
                ci_percentile=float(args.all_data_ci_level),
                stabilization_ci_start=args.stabilization_ci_start,
                bins=args.bins,
                output_dir=output_subdir,
            )
        else:
            stage3_results = bootstrap_stratified_cv_modeling(
                bootstrapped_data=bootstrapped_data_stage3,
                perturbed_tf_series=input_data.predictors_df[input_data.perturbed_tf],
                estimator=estimator,
                ci_percentiles=[float(args.all_data_ci_level)],
                bins=args.bins,
            )

        # 6. Serialize and save the Stage 3 results
        stage3_output_dir = os.path.join(output_subdir, "stage3_result_object")
        os.makedirs(stage3_output_dir, exist_ok=True)
        stage3_results.serialize("result_obj", stage3_output_dir)

        stage3_sig_coefs = stage3_results.extract_significant_coefficients(
            ci_level=args.all_data_ci_level,
        )

        stage3_ci_str = str(args.all_data_ci_level).replace(".", "-")
        stage3_output_file = os.path.join(
            output_subdir, f"stage3_2b_significant_{stage3_ci_str}.json"
        )
        logger.info(f"Writing Stage 3 2b significant results to {stage3_output_file}")
        with open(stage3_output_file, "w") as f:
            json.dump(stage3_sig_coefs, f, indent=4)

    # ==========================================
    # END NEW CODE
    # ==========================================

    logger.info(
        "Step 5: Test the significance of the interactor terms that survive "
        "against the corresponding main effect"
    )

    if args.stage4_topn:
        logger.info("Stage 4 will use top-n masked input data.")
        input_data.top_n_masked = True
    else:
        logger.info("Stage 4 will use full input data.")

    # calculate the statification classes for the perturbed TF (all data)
    stage4_classes = stratification_classification(
        input_data.predictors_df[input_data.perturbed_tf].squeeze(),
        bins=args.bins,
    )

    # Test the significance of the interactor terms
    evaluate_interactor_significance = (
        evaluate_interactor_significance_lassocv
        if args.stage4_lasso
        else evaluate_interactor_significance_linear
    )

    results = evaluate_interactor_significance(
        input_data,
        stratification_classes=stage4_classes,
        model_variables=list(
            topn_results.extract_significant_coefficients(
                ci_level=args.topn_ci_level
            ).keys()
        ),
        estimator=estimator,
    )

    output_significance_file = os.path.join(
        output_subdir, "interactor_vs_main_result.json"
    )
    logger.info(
        "Writing the final interactor significance "
        f"results to {output_significance_file}"
    )
    results.serialize(output_significance_file)


def parse_bins(s):
    try:
        return [np.inf if x == "np.inf" else int(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid bin value in '{s}'")


def parse_comma_separated_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_json_dict(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")


# Allowed keys for method='L-BFGS-B' (excluding deprecated options)
LBFGSB_ALLOWED_KEYS = {
    "maxcor",  # int
    "ftol",  # float
    "gtol",  # float
    "eps",  # float or ndarray
    "maxfun",  # int
    "maxiter",  # int
    "maxls",  # int
    "finite_diff_rel_step",  # float or array-like or None
}


def parse_lbfgsb_options(s):
    try:
        opts = json.loads(s)
        if not isinstance(opts, dict):
            raise ValueError("Options must be a JSON object")

        unexpected_keys = set(opts) - LBFGSB_ALLOWED_KEYS
        if unexpected_keys:
            raise argparse.ArgumentTypeError(
                f"Unexpected keys in --minimize_options: {unexpected_keys}"
            )
        return opts
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def add_general_arguments_to_subparsers(subparsers, general_arguments):
    for subparser in subparsers.choices.values():
        for arg in general_arguments:
            subparser._add_action(arg)


def common_modeling_binning_arguments(parser: argparse._ArgumentGroup) -> None:
    parser.add_argument(
        "--bins",
        type=parse_bins,
        default="0,8,64,512,np.inf",
        help=(
            "Comma-separated list of bin edges (integers or 'np.inf'). "
            "Default is --bins 0,8,12,np.inf"
        ),
    )


def common_modeling_input_arguments(
    parser: argparse._ArgumentGroup, top_n_default: int | None = 600
) -> None:
    """Add common input arguments for modeling commands."""
    parser.add_argument(
        "--response_file",
        type=str,
        required=True,
        help=(
            "Path to the response CSV file. The first column must contain "
            "feature names or locus tags (e.g., gene symbols), matching the index "
            "format in both response and predictor files. The perturbed gene will "
            "be removed from the model data only if its column names match the "
            "index format."
        ),
    )
    parser.add_argument(
        "--predictors_file",
        type=str,
        required=True,
        help=(
            "Path to the predictors CSV file. The first column must contain "
            "feature names or locus tags (e.g., gene symbols), ensuring consistency "
            "between response and predictor files."
        ),
    )
    parser.add_argument(
        "--perturbed_tf",
        type=str,
        required=True,
        help=(
            "Name of the perturbed transcription factor (TF) used as the "
            "response variable. It must match a column in the response file."
        ),
    )
    parser.add_argument(
        "--blacklist_file",
        type=str,
        default="",
        help=(
            "Optional file containing a list of features (one per line) to be excluded "
            "from the analysis."
        ),
    )
    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap samples to generate for resampling. Default is 1000",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Set this to an integer to make the bootstrap sampling reproducible. "
        "Default is None (no fixed seed) and each call will produce different "
        "bootstrap indices. Note that if this is set, the `top_n` random_state will "
        "be +10 in order to make the top_n indices different from the `all_data` step",
    )
    parser.add_argument(
        "--normalize_sample_weights",
        action="store_true",
        help=(
            "Set this to normalize the sample weights to sum to 1. " "Default is False."
        ),
    )
    parser.add_argument(
        "--scale_by_std",
        action="store_true",
        help=(
            "Set this to scale the model matrix by standard deviation"
            "(without centering). The data is scaled using"
            "StandardScaler(with_mean=False, with_std=True). The estimator will"
            "still fit an intercept (fit_intercept=True) since the "
            "data is not centered."
        ),
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=top_n_default,
        help=(
            "Number of features to retain in the second round of modeling. "
            f"Default is {top_n_default}"
        ),
    )


def common_modeling_feature_options(parser: argparse._ArgumentGroup) -> None:
    parser.add_argument(
        "--row_max",
        action="store_true",
        help=(
            "Include the row max as an additional predictor in the model matrix "
            "in the first round (all data) model."
        ),
    )
    parser.add_argument(
        "--squared_pTF",
        action="store_true",
        help=(
            "Include the squared pTF as an additional predictor in the model matrix "
            "in the first round (all data) model."
        ),
    )
    parser.add_argument(
        "--cubic_pTF",
        action="store_true",
        help=(
            "Include the cubic pTF as an additional predictor in the model matrix "
            "in the first round (all data) model."
        ),
    )
    parser.add_argument(
        "--exclude_interactor_variables",
        type=parse_comma_separated_list,
        default=[],
        help=(
            "Comma-separated list of variables to exclude from the interactor terms. "
            "E.g. red_median,green_median. To exclude all variables, use 'exclude_all'"
        ),
    )
    parser.add_argument(
        "--add_model_variables",
        type=parse_comma_separated_list,
        default=[],
        help=(
            "Comma-separated list of variables to add to the all_data model. "
            "E.g., red_median,green_median would be added as ... + red_median + "
            "green_median"
        ),
    )
    parser.add_argument(
        "--ptf_main_effect",
        action="store_true",
        help=(
            "Include the perturbed transcription factor (pTF) main effect in the "
            "modeling formula. This is added to the all_data model formula."
        ),
    )
    parser.add_argument(
        "--stage2_set_zero",
        action="store_true",
        help=(
            "Set all non-top LRB interactions to be zero in the stage 2 model."
        ),
    )

    parser.add_argument(
        "--skip_1st_stage",
        action="store_true",
        help=(
            "Skip the first stage of modeling and go directly to the second stage."
        ),
    )

    parser.add_argument(
        "--stage3_2b",
        dest="stage3_2b",
        action="store_true",
        help=(
            "For stage 3, rerun the stage 1 bootstrap with independent and interactor effects from stage 2"
        ),
    )

