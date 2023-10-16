import numpy as np
import random
import math

# Constants
INTERACTION_MATRIX = {
    ('P', 'P'): -2,
    ('P', 'H'): 1,
    ('H', 'P'): 1,
    ('H', 'H'): -3
}

SOLVENT_INTERACTIONS_CORNERS = {
    'P': -2,
    'H': 2
}

SOLVENT_INTERACTIONS_SIDES = {
    'P': -1,
    'H': 0
}

def generate_sequence(num_residue):
    return [random.choice(['H', 'P']) for _ in range(num_residue)]

def calculate_energy(sequence, nc_seq_pairs, matrix):
    energy = 0
    for pair in nc_seq_pairs:
        interaction = INTERACTION_MATRIX.get(pair, 0)
        energy += interaction

    # Add solvent interactions based on the sequence
    corners, exposed_sides = get_matrix_corners_and_sides(matrix)
    for i in corners:
        energy += SOLVENT_INTERACTIONS_CORNERS[sequence[i - 1]]
    for i in exposed_sides:
        energy += SOLVENT_INTERACTIONS_SIDES[sequence[i - 1]]

    return energy

def mutate_sequence(sequence):
    # Clone the original sequence
    mutated_sequence = sequence[:]

    # Choose two random positions to swap
    pos1, pos2 = random.sample(range(len(mutated_sequence)), 2)

    # Swap the residues at the selected positions
    mutated_sequence[pos1], mutated_sequence[pos2] = mutated_sequence[pos2], mutated_sequence[pos1]

    return mutated_sequence


def metropolis_criterion(current_energy, proposed_energy, temperature):
    if proposed_energy < current_energy:
        return True
    else:
        delta_energy = proposed_energy - current_energy
        probability = math.exp(-delta_energy / temperature)
        return random.random() < probability

def monte_carlo_simulation(matrix, temperature, num_steps):
    num_residue = matrix.size
    best_sequence = generate_sequence(num_residue)
    best_energy = calculate_energy(best_sequence, get_neighbours(matrix, best_sequence), matrix)

    current_sequence = best_sequence
    current_energy = best_energy

    for step in range(num_steps):
        # Propose a sequence change (e.g., swap two residues)
        # Implement your method to change the sequence (e.g., swap two residues)
        new_sequence = mutate_sequence(current_sequence)
        # Update proposed_energy based on the new_sequence
        proposed_energy = calculate_energy(new_sequence, get_neighbours(matrix, new_sequence), matrix)

        # Check if we accept the proposed sequence change
        if metropolis_criterion(current_energy, proposed_energy, temperature):
            current_sequence = new_sequence
            current_energy = proposed_energy

            # Update the best sequence and energy if necessary
            if proposed_energy < best_energy:
                best_sequence = current_sequence
                best_energy = proposed_energy

    return best_sequence, best_energy

def get_matrix_corners_and_sides(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    if num_rows < 2 or num_cols < 2:
        return [], []

    corners = [matrix[0][0], matrix[0][num_cols - 1], matrix[num_rows - 1][0], matrix[num_rows - 1][num_cols - 1]]

    exposed_sides = []
    for i in range(num_rows):
        for j in range(num_cols):
            if i == 0 and j not in [0, num_cols - 1]:
                exposed_sides.append(matrix[i][j])
            elif i == num_rows - 1 and j not in [0, num_cols - 1]:
                exposed_sides.append(matrix[i][j])
            elif j == 0 and i not in [0, num_rows - 1]:
                exposed_sides.append(matrix[i][j])
            elif j == num_cols - 1 and i not in [0, num_rows - 1]:
                exposed_sides.append(matrix[i][j])

    return corners, exposed_sides

def get_neighbours(matrix, sequence):
    # Initialize the list to store the pairs and a set to keep track of seen pairs
    pairs = []
    seen_pairs = set()
    for i in range(4):
        for j in range(4):
            element_1 = matrix[i][j]

            # Define the possible relative positions for element_2
            positions = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                        (i, j - 1), (i, j + 1), (i + 1, j - 1),
                        (i + 1, j), (i + 1, j + 1)]

            # Filter out positions that are outside the matrix
            valid_positions = [(x, y) for x, y in positions if 0 <= x < 4 and 0 <= y < 4]

            # Create pairs with element_1 and the valid elements
            for x, y in valid_positions:
                element_2 = matrix[x][y]
                pair = (element_1, element_2)

                # Check if the pair has already been added
                if pair not in seen_pairs and (pair[1], pair[0]) not in seen_pairs:
                    pairs.append(pair)
                    seen_pairs.add(pair)
    non_covalent_pairs = [pair for pair in pairs if abs(pair[0] - pair[1]) != 1]     # filtering out the covalent interactions
    nc_seq_pairs = [(sequence[i - 1], sequence[j - 1]) for i, j in non_covalent_pairs]  # converting the structure positions to the sequence
    return nc_seq_pairs

def main():
    matrix = np.array([  
        [1, 2, 15, 16],
        [4, 3, 14, 13],
        [5, 8, 9, 12],
        [6, 7, 10, 11]
    ])
    
    num_residue = matrix.size
    corners, exposed_sides = get_matrix_corners_and_sides(matrix)
    
    temperature = 1.0  # Adjust the temperature
    num_steps = 1000  # Number of Monte Carlo steps

    best_sequence, best_energy = monte_carlo_simulation(matrix, temperature, num_steps)
    
    print("Best Sequence:", best_sequence)
    print("Best Energy:", best_energy)

if __name__ == "__main__":
    main()
