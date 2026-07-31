
import pandas as pd
from sklearn.linear_model import LinearRegression


# Weighted scoring system
DESCRIPTION_WEIGHTS = {
    'swinging_strike': 1.0,
    'swinging_strike_blocked': 1.0,
    'called_strike': 0.7,
    'foul': 0.3,
    'foul_tip': 0.3,
    'ball': 0.0,
    'ball_in_dirt': 0.0,
    'hit_by_pitch': -1.0,
    'blocked_ball': -0.2,
    'wild_pitch': -1.0,
    'in_play': 0.2  # will be overridden for weak contact
}

FEATURE_COLUMNS = [
    'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z'
]


def calculate_stuff_plus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Stuff+ scores for each pitch in a Statcast dataframe.
    Returns the same dataframe with `stuff_score` and `Stuff+` columns added.
    """
    df = df.copy()

    # Ensure necessary features exist
    if not all(col in df.columns for col in FEATURE_COLUMNS):
        raise ValueError(f"Missing one or more required columns: {FEATURE_COLUMNS}")

    # Drop null rows
    df.dropna(subset=FEATURE_COLUMNS + ['description'], inplace=True)

    # Assign outcome weights
    df['stuff_score'] = df['description'].map(DESCRIPTION_WEIGHTS).fillna(0.2)

    # Reward weak contact: in_play + low launch_speed
    if 'launch_speed' in df.columns:
        weak_contact = (df['description'] == 'in_play') & (df['launch_speed'] < 85)
        df.loc[weak_contact, 'stuff_score'] = 0.4

    # Only proceed if there is enough data
    if len(df) < 1:
        print("Warning: Not enough pitch data to train Stuff+ model. Returning neutral scores.")
        df['Stuff+'] = 100.0
        return df

    # Train model
    X = df[FEATURE_COLUMNS]
    y = df['stuff_score']
    model = LinearRegression().fit(X, y)

    # Predict stuff_score from features
    df['stuff_score'] = model.predict(X)

    # Normalize to get Stuff+
    league_avg = df['stuff_score'].mean() or 1
    df['Stuff+'] = (df['stuff_score'] / league_avg) * 100

    return df
