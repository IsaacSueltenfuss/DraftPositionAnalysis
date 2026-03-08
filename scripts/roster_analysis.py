import nflreadpy as nfl
import pandas as pd
import numpy as np

def getRosterData(team_list):
    # Load Data
    rosters_2025 = nfl.load_rosters([2025]).to_pandas()
    rosters_2025 = rosters_2025[rosters_2025['team'].isin(team_list)]
    snaps_raw = nfl.load_snap_counts([2025]).to_pandas()
    player_map = nfl.load_players().to_pandas()

    pos_map = {
        'HB': 'RB',
        'OT': 'OL', 'OG': 'OL', 'G': 'OL', 'T': 'OL', 'C': 'OL',        
        'DE': 'DL', 'DT': 'DL', 'NT': 'DL',
        'OLB': 'LB', 'MLB': 'LB', 'ILB': 'LB',
        'CB': 'DB', 'S': 'DB', 'FS': 'DB', 'SS': 'DB', 'SAF': 'DB',
    }
    offense_pos = ['QB', 'RB', 'FB', 'WR', 'TE', 'OL']
    defense_pos = ['DL', 'LB', 'DB']

    snaps = snaps_raw[snaps_raw['game_type'] == 'REG'].copy()
    snaps['position'] = snaps['position'].replace(pos_map)

    def get_unit_snap(row):
        if row['position'] in offense_pos:
            return row.get('offense_pct', 0)
        elif row['position'] in defense_pos:
            return row.get('defense_pct', 0)
        return row.get('st_pct', 0)

    snaps['clean_snap'] = snaps.apply(get_unit_snap, axis=1)
    snaps = snaps.merge(player_map[['pfr_id', 'gsis_id']], left_on='pfr_player_id', right_on='pfr_id')

    # Aggregates games over 17-game season
    player_season_snaps = snaps.groupby('gsis_id', as_index=False).agg(
        snap_pct=('clean_snap', lambda x: x.sum() / 17)
    )

    df = rosters_2025.merge(player_season_snaps, on='gsis_id', how='left')
    df = df.merge(
        player_map[['gsis_id', 'draft_round']], 
        on='gsis_id', 
        how='left'
    )
    df['snap_pct'] = df['snap_pct'].fillna(0)

    df['position'] = df['position'].replace(pos_map)
    pos_order = ['QB', 'RB', 'FB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K', 'P', 'LS']
    df = df.dropna(subset=['position'])
    df['position'] = pd.Categorical(df['position'], categories=pos_order, ordered=True)
    df['draft_number'] = df['draft_number'].fillna(260)
    df['draft_round'] = df['draft_round'].fillna(8)
    df = df.drop_duplicates(subset=['gsis_id'])

    def weighted_stats(group):
        total_snaps = group['snap_pct'].sum()
        if total_snaps == 0:
            return pd.Series({
                'weighted_pick': group['draft_number'].mean(),
                'weighted_round': group['draft_round'].mean(),
                'player_count': float(group['gsis_id'].nunique())
            })
        
        d_num = (group['draft_number'] * group['snap_pct']).sum() / total_snaps
        d_rnd = (group['draft_round'] * group['snap_pct']).sum() / total_snaps
        
        return pd.Series({
            'weighted_pick': d_num,
            'weighted_round': d_rnd,
            'player_count': float(group['gsis_id'].nunique())
        })

    final_summary = df.groupby(['team', 'position'], observed=True).apply(
        lambda x: weighted_stats(x)
    ).reset_index()
    final_summary = final_summary.sort_values(['team', 'weighted_round'])

    return final_summary

if __name__ == "__main__":
    nfc_east = ["DAL", "PHI", "NYG", "WAS"]
    results_df = getRosterData(nfc_east)
    
    results_df.to_csv("../data/nfc_east_rosters.csv", index=False)
    print(results_df.head(10))