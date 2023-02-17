from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

class Database:
    def __init__(self, database: pd.DataFrame, preprocessors: Dict[str, Callable] = {}) -> None:
        # Set database and define column names
        self._ensure_dataframe(database)
        self.columns = database.columns.tolist()

        # Set and perform preprocessing
        self.preprocessors = self._ensure_preprocessors(preprocessors)
        self.database = self._preprocess_dataframe(database)


    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame([])

        for col in df:
            preproc = self.preprocessors.get(col, lambda x: str(x).lower())
            result[col] = [preproc(x) for x in df[col]]

        return result


    def _preprocess_record(self, record: Dict[str, str]) -> Dict[str, str]:
        result = {}

        for col, val in record.items():
            preproc = self.preprocessors.get(col, lambda x: str(x).lower())
            result[col] = preproc(val)

        return result


    def _ensure_preprocessors(self, preps: Dict[str, Callable]) -> Dict[str, Callable]:
        for k, v in preps.items():
            if not callable(v):
                raise TypeError(f'Preprocessor for field {k} is not a callable')
            if k not in self.columns:
                raise ValueError(f'Column {k} does not exist in database, yet a preprocessor for it is defined')

        return preps


    def _ensure_dataframe(self, obj) -> pd.DataFrame:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError('Must supply a pandas dataframe')

        return obj


    def _ensure_scorers(self, scorers: Dict[str, Callable], record: pd.DataFrame) -> Dict[str, Callable]:
        for k, v in scorers.items():
            if not callable(v):
                raise TypeError(f'Scorer for field {k} is not a callable')
            if k not in self.columns:
                raise ValueError(f'Column {k} does not exist in database, yet a scorer for it is defined')
            if k not in record:
                raise ValueError(f'Column {k} does not exist in record, yet a scorer for it is defined')

        return scorers


    def _ensure_record(self, record) -> Dict[str, str]:
        if not any([x in self.columns for x in record]):
            raise ValueError('Record does not contain any columns in database')
        if not isinstance(record, dict):
            raise TypeError('Single record must be supplied in dict format')

        return record


    def _score_one_field(self, field, column, scorer: Optional[Callable] = None) -> List[float]:
        # NOTE: field shall be single unit, not multiple at once
        if scorer is None:
            scorer = fuzz.ratio

        column_score = process.cdist([column], self.database[field], scorer=scorer)

        # NOTE: because field shall be single unit and cdist give a (1, shape) shape,
        # A little fix is done here so that the returned value can be converted to a Series
        # i.e. 1D-array. Using subscript[0] instead of .ravel() is intentional to show that this is a problem.
        # FIXME: This shouldn't be like this. If possible, fix using better design decision.
        return column_score[0]


    def calculate_scores(self,
                         record: Dict[str, str],
                         scorers: Dict[str, Callable] = {},
                         summary_method: str = 'geom_mean',
                         weights: Optional[List[float]] = None) -> pd.DataFrame:
        """Calculate score for record against all items in the database.

        Args:
            record (Dict[str, str]): Record to score against.
            scorers (Dict[str, Callable], optional): Scorer functions. Defaults to {}.
            summary_method (str, optional): Summary methods: {'geom_mean', 'mean'}. Defaults to 'geom_mean'.
            weights (Optional[List[float]], optional): Weight used in calculating summary (average). Defaults to None.

        Returns:
            pd.DataFrame: Score for record against all items in the database.
        """
        # Perform checks
        record = self._ensure_record(record)
        scorers = self._ensure_scorers(scorers, record)

        # Preprocess
        record = self._preprocess_record(record)

        # Column-wise match score
        result = pd.DataFrame([])
        for col in record:
            if col in self.columns:
                result[col] = self._score_one_field(col, record[col], scorers.get(col))

        # Summarize into overall score
        if summary_method == 'geom_mean':
            result['overall'] = np.exp(np.average(np.log(result), axis=1, weights=weights))
        elif summary_method == 'mean':
            result['overall'] = np.average(result, axis=1, weights=weights)
        else:
            # TODO: What if user wants to give weights in Dict[str, float] format?
            # TODO: What if user wants some custom summarization method?
            raise ValueError(f'Unrecognized value "{summary_method}" for summary_method argument.')

        return result


    def get_match(self, record: Dict[str, str], *,
                        scorers: Dict[str, Callable] = {},
                        n_match: int = 1,
                        summary_method: str = 'geom_mean',
                        weights: Optional[List[float]] = None) -> pd.DataFrame:
        scores = self.calculate_scores(record, scorers, summary_method, weights)
        scores.columns = [f'score_{x}' for x in scores.columns]


        result = pd.concat([
            scores,
            self.database.copy(), # Inefficient
        ], axis=1).sort_values('score_overall', ascending=False).iloc[:n_match, :]

        result = result.rename(columns={k: f"db_{k}" for k in record})

        # List [v] is used instead of scalar v, to avoid this:
        # Exception has occurred: ValueError
        # If using all scalar values, you must pass an index
        record_ = pd.DataFrame({f"rec_{k}": [v]*len(result) for k, v in record.items()})
        record_ = record_.reset_index(drop=True)

        result = pd.concat([
            record_,
            result.reset_index()
        ], axis=1)

        return result
