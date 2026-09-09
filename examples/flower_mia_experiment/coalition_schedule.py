import random

def build_coalition_schedule(client_ids, target_id, rounds, coalition_size):
    """
    Return a list of coalitions, balanced so half include target_id and half exclude it.
    """
    schedule = []
    other_clients = [c for c in client_ids if c != target_id]
    
    for i in range(rounds):
        if i % 2 == 0:
            # Include target_id
            coalition = [target_id] + random.sample(other_clients, coalition_size - 1)
        else:
            # Exclude target_id
            coalition = random.sample(other_clients, coalition_size)
            
        random.shuffle(coalition)
        schedule.append(coalition)
        
    # Shuffle schedule so positive/negative runs are interleaved randomly
    random.shuffle(schedule)
    return schedule
