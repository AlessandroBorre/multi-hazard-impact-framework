import numpy as np

def recovery_velocity_shape():
    """
    Randomly select a recovery vector from predefined options or allow user input.

    Returns:
    tuple: A tuple with response time, recovery time, and curve type.
    """
    use_random = input("Do you want to use random recovery parameters? (yes/no): ").strip().lower()

    if use_random == 'yes':
        options = np.array(
            [[30, 150, 'Log'], [70, 230, 'Log'], [110, 300, 'Lin'], [200, 450, 'Exp'], [150, 380, 'Exp']], dtype=object)
        selected_option = options[np.random.choice(options.shape[0])]
        response_time, recovery_time, curve_type = selected_option
    else:
        response_time = int(input("Enter the response time for 100% damage (in days): "))
        recovery_time = int(input("Enter the recovery time for 100% damage (in days): "))
        curve_type = input("Enter the recovery equation type (Lin, Exp, Log): ").strip()

    return response_time, recovery_time, curve_type


def funcResp(y, mResp, qResp):
    """
    Function that calculates the response time for each event based on the damage intensity.
    """
    return (y - qResp) / mResp


def funcRec(y, mR, qR):
    """
    Function that calculates the recovery time for each event based on the damage intensity.
    """
    return (y - qR) / mR


def R_linear(t, D, t_startrec, t_endrec):
    """
    Calculates the linear recovery function R(t).
    Parameters:
    t (float): Current time
    D (float): Initial damage
    t_startrec (float): Recovery start time
    t_endrec (float): Recovery end time
    Returns:
    float: Value of R(t) between 0 and 1
    """
    if t < t_startrec:
        return 0
    elif t >= t_endrec:
        return 1
    else:
        return 1- (t - t_startrec) / (t_endrec - t_startrec)


def funcRecExp(t, D, t_startrec, t_endrec):
    """
    Calcola la funzione di recupero esponenziale corretta, che inizia velocemente e rallenta.

    Parameters:
    t (float): Tempo corrente
    D (float): Danno iniziale
    t_startrec (float): Tempo di inizio del recupero
    t_endrec (float): Tempo di fine del recupero

    Returns:
    float: Valore di R(t) tra 0 e 1
    """
    if t < t_startrec:
        return 0
    elif t >= t_endrec:
        return 1
    else:
        # Normalizziamo il tempo tra t_startrec e t_endrec
        recovery_rate = (t - t_startrec) / (t_endrec - t_startrec)
        # Aggiungiamo una piccola costante epsilon per lisciare l'inizio
        epsilon = 1e-40  # Soglia piccola per evitare lo scalino
        k = 6.4  # Manteniamo k regolabile
        return 1 - np.exp(-k * (1 - (recovery_rate + epsilon)))


def funcRecLog(t, D, t_startrec, t_endrec):
    """
    Calcola la funzione di recupero logaritmica R(t).

    Parameters:
    t (float): Tempo corrente
    D (float): Danno iniziale
    t_startrec (float): Tempo di inizio del recupero
    t_endrec (float): Tempo di fine del recupero

    Returns:
    float: Valore di R(t) tra 0 e 1
    """
    if t < t_startrec:
        return 0
    elif t >= t_endrec:
        return 1
    else:
        # Progresso normalizzato tra 0 e 1
        recovery_rate = (t - t_startrec) / (t_endrec - t_startrec)
        # Calcoliamo il recupero logaritmico invertito (che parte velocemente e rallenta alla fine)
        return 1 - (np.log(1 + recovery_rate * 9) / np.log(10))

# Note:
# These functions calculate response and recovery times and the recovery function R(t).
# They can be adjusted or extended to support additional recovery equations or parameters.
