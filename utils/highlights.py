import requests

def get_game_highlight_plays(game_pk):
    """
    Pull highlight clips from a specific MLB game.
    Returns a list of dicts with playId, title, and URL.
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    r = requests.get(url)
    if not r.ok:
        return []

    data = r.json()
    highlights = []

    for play in data['liveData']['plays']['allPlays']:
        media = play.get('media', {})
        if 'epg' in media:
            for group in media['epg']:
                if group.get('title') == 'highlight':
                    for item in group.get('items', []):
                        play_id = item.get('playId')
                        title = item.get('title')
                        player = item.get('player', {}).get('fullName', '')
                        url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
                        highlights.append({
                            'title': title,
                            'player': player,
                            'url': url,
                            'play_id': play_id
                        })
    return highlights
