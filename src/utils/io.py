import pandas as pd
import os

def load_df(path: str) -> pd.DataFrame:
    """Load parquet file from data/raw/ or specified path."""
    if not path.startswith("raw/") and not os.path.exists(path):
        path = f"raw/{path}"
    if not path.endswith(".parquet"):
        path = f"{path}.parquet"
    
    df = pd.read_parquet(path)
    
    # Ensure index is DatetimeIndex if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            pass
    
    # Ensure column names are properly capitalized
    df.columns = [col.title() if isinstance(col, str) else col for col in df.columns]
    
    # Ensure we have required columns
    required = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing required columns in {path}. Found: {df.columns.tolist()}")
    
    return df

