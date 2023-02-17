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
        self.database = self._preprocess_columns(database)
    

    def _preprocess_columns(self, df: pd.DataFrame) -> None:
        def default_preproc(x):
            return str(x).lower()
        
        for col in df:
            preproc = self.preprocessors.get(col, default_preproc)
            df[col] = df[col].apply(lambda x: preproc(x))
        
        return df
    

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
    

    def _ensure_columns(self, record: pd.DataFrame) -> pd.DataFrame:
        if not any([x in self.columns for x in record]):
            raise ValueError('Record does not contain any columns in database')
        
        return record

    
    def _ensure_scorers(self, scorers: Dict[str, Callable], record: pd.DataFrame) -> Dict[str, Callable]:
        for k, v in scorers.items():
            if not callable(v):
                raise TypeError(f'Scorer for field {k} is not a callable')
            if k not in self.columns:
                raise ValueError(f'Column {k} does not exist in database, yet a scorer for it is defined')
            if k not in record:
                raise ValueError(f'Column {k} does not exist in record, yet a scorer for it is defined')
        
        return scorers


    def _column_match(self, field, column, scorer: Optional[Callable] = None) -> List[float]:
        # NOTE: field shall be single unit, not multiple at once
        if scorer is None:
            scorer = fuzz.ratio
        
        column_score = process.cdist([column], self.database[field], scorer=scorer)

        # NOTE: because field shall be single unit and cdist give a (1, shape) shape,
        # A little fix is done here so that the returned value can be converted to a Series
        # i.e. 1D-array. Using subscript[0] instead of .ravel() is intentional to show that this is a problem.
        # FIXME: This shouldn't be like this. If possible, fix using better design decision.
        return column_score[0]

    
    def calculate_overall_score(self,
                                record: pd.DataFrame,
                                scorers: Dict[str, Callable] = {},
                                summary_method: str = 'geom_mean',
                                weights: Optional[List[float]] = None) -> pd.DataFrame:
        # Perform checks
        record = self._ensure_columns(record)
        scorer = self._ensure_scorers(scorers, record)

        # Preprocess
        record = self._preprocess_columns(record)

        # Column-wise match score
        result = pd.DataFrame([])
        for col in record:
            if col in self.columns:
                result[col] = self._column_match(col, record.at[0, col], scorer.get(col))
        
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
    

    # FIXME, NOTE, TODO:
    # So far, this method only gives out n_matches relevant records.
    # This is already helpful because it gives context, but still feels wonky wobbly somehow.
    def find_match(self, record: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
        # Dirty trick because I don't support multiple records
        if len(record) > 1:
            raise ValueError('Please supply dataframe of length one as record')
        
        scores = self.calculate_overall_score(record)
        scores.columns = [f'score_{x}' for x in scores.columns]


        result = pd.concat([
            self.database.copy(), # Inefficient
            scores
        ], axis=1).sort_values('score_overall', ascending=False).iloc[:n_matches, :]
        
        return result
