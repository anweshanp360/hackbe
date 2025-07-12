# matching_logic.py

def match_trials(posted_data, trials_data):
    """
    Compares each posted item against trials and returns matching results.
    Matching Rule: name matches and value >= min_value
    """
    if not trials_data or not posted_data:
        return []

    matching_results = []

    for posted_item in posted_data:
        posted_name = posted_item.get('name')
        posted_value = posted_item.get('value')

        if posted_name is None or posted_value is None:
            continue

        matches = [
            trial for trial in trials_data
            if trial.get('name') and trial.get('min_value') is not None
            and trial['name'].lower() == posted_name.lower()
            and posted_value >= trial['min_value']
        ]

        if matches:
            matching_results.append({
                "posted_data": posted_item,
                "matches": matches
            })

    return matching_results
