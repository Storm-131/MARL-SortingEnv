# ---------------------------------------------------------*\
# Title: Seasonal Inputs (Overview)
# ---------------------------------------------------------*/
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------*/
# Seasonal Input Generator
# ---------------------------------------------------------*/

class SeasonalInputGenerator:
    """Generates seasonal input patterns for the sorting system with four materials."""

    def __init__(self, seed=None, steps_per_pattern=20):
        self.material_names = ['A', 'B', 'C', 'D']
        self.patterns = {
            1: {'A': 0.40, 'C': 0.35, 'B': 0.15, 'D': 0.10}, # A & C dominant
            2: {'B': 0.40, 'D': 0.35, 'A': 0.15, 'C': 0.10}, # B & D dominant
        }
        self.current_pattern_idx = 0
        self.steps_per_pattern = steps_per_pattern
        self.step_counter = 0
        self.set_seed(seed)

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Shuffle the order of patterns for this run
        self.pattern_sequence = self.rng.permutation(list(self.patterns.keys()))

    def select_pattern(self):
        """Selects the next pattern in the shuffled sequence."""
        self.current_pattern_idx = (self.current_pattern_idx + 1) % len(self.pattern_sequence)
        self.step_counter = 0

    def generate_input(self, batchsize=100):
        """
        Generates a batch of units based on the current pattern.
        Switches to the next pattern every `steps_per_pattern` calls.
        """
        if self.step_counter >= self.steps_per_pattern:
            self.select_pattern()

        pattern_key = self.pattern_sequence[self.current_pattern_idx]
        ratios = self.patterns[pattern_key]

        # Calculate units per material for the given batchsize
        units = {mat: int(np.floor(ratios[mat] * batchsize)) for mat in self.material_names}
        
        # Distribute rounding errors
        remainder = batchsize - sum(units.values())
        for _ in range(remainder):
            # Add to a random material
            units[self.rng.choice(self.material_names)] += 1

        # Create and shuffle the batch
        batch = []
        for mat, count in units.items():
            batch.extend([mat] * count)
        self.rng.shuffle(batch)
        
        self.step_counter += 1
        return batch

    # ---------------------------------------------------------*/
    # Plotting Functions
    # ---------------------------------------------------------*/
    def plot_seasonal_inputs(self):
        """Plot the seasonal input patterns over time."""
        self.set_seed(self.seed)
        total_units = 4000
        units_generated = 0
        materials = []
        patterns = []
        while units_generated < total_units:
            # Generate units until we have 4000 units
            units_left = total_units - units_generated
            units_to_generate = self.total_units - self.current_position
            if units_to_generate == 0:
                # Need to generate a new sample and select a new pattern
                self.select_pattern()
                units_to_generate = min(units_left, self.total_units - self.current_position)
            else:
                units_to_generate = min(units_left, units_to_generate)
            materials.extend(self.materials[self.current_position:self.current_position+units_to_generate])
            patterns.extend([self.current_pattern]*units_to_generate)
            self.current_position += units_to_generate
            units_generated += units_to_generate

        # Now we have materials and patterns lists of length 4000
        # The window size for moving averages defines the smoothing level
        window_size = 100  # You can adjust the window size as needed
        
        a_values = np.convolve([1 if m == 'A' else 0 for m in materials],
                               np.ones(window_size)/window_size, mode='valid')
        b_values = np.convolve([1 if m == 'B' else 0 for m in materials],
                               np.ones(window_size)/window_size, mode='valid')
        c_values = np.convolve([1 if m == 'C' else 0 for m in materials],
                               np.ones(window_size)/window_size, mode='valid')
        d_values = np.convolve([1 if m == 'D' else 0 for m in materials],
                               np.ones(window_size)/window_size, mode='valid')
        e_values = np.convolve([1 if m == 'E' else 0 for m in materials],
                               np.ones(window_size)/window_size, mode='valid')
        # Adjust patterns list to match the length of the moving averages
        patterns = patterns[window_size-1:]
        self.plot_inputs(a_values, b_values, c_values, d_values, e_values,
                         patterns, 'Seasonal Input Patterns Over Time (4000 Units)')

    def plot_inputs(self, a_values, b_values, c_values, d_values, e_values, patterns, title):
        """Plot the seasonal input patterns."""

        sns.set_theme()
        sns.set_style("white")
        plt.figure(figsize=(15, 8))
        x = np.arange(len(a_values))
        plt.plot(x, a_values, label='Material A', linestyle='-', color='blue')
        plt.plot(x, b_values, label='Material B', linestyle='-', color='red')
        plt.plot(x, c_values, label='Material C', linestyle='-', color='green')
        plt.plot(x, d_values, label='Material D', linestyle='-', color='purple')
        plt.plot(x, e_values, label='Material E', linestyle='-', color='orange')

        # Shade background according to patterns
        colors = ['orange', 'cyan'] # Colors for the two patterns

        pattern_labels = {
            1: "Pattern 1: A & C dominant",
            2: "Pattern 2: B & D dominant",
        }

        current_start = 0
        current_pattern = patterns[0]
        for i in range(1, len(patterns)):
            if patterns[i] != current_pattern:
                plt.axvspan(current_start, i, color=colors[current_pattern-1], alpha=0.2)
                current_start = i
                current_pattern = patterns[i]
        plt.axvspan(current_start, len(patterns), color=colors[current_pattern-1], alpha=0.2)

        # Create custom legend for line colors (Materials)
        material_lines = [
            plt.Line2D([0], [0], color='blue', lw=2),
            plt.Line2D([0], [0], color='red', lw=2),
            plt.Line2D([0], [0], color='green', lw=2),
            plt.Line2D([0], [0], color='purple', lw=2),
            plt.Line2D([0], [0], color='orange', lw=2)
        ]
        # Custom legend for background colors (Patterns)
        pattern_patches = [
            plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.2)
            for i in range(len(self.patterns))
        ]
        # Combine legends
        legend1 = plt.legend(material_lines, ['Material A', 'Material B',
                             'Material C', 'Material D', 'Material E'], loc='upper left')
        legend2 = plt.legend(pattern_patches, [pattern_labels[i+1] for i in range(len(self.patterns))], loc='upper right')
        plt.gca().add_artist(legend1)

        fontsize = 16
        plt.xlabel('Unit Number', fontsize=fontsize)
        plt.ylabel('Proportion', fontsize=fontsize)
        plt.title(title, fontsize=20, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(f'seasonal_input_patterns_seed_{self.seed}.svg')
        plt.show()

    def plot_ordered_patterns(self):
        """Plot all four patterns in ascending order."""
        self.set_seed(self.seed)
        total_units_per_pattern = 1000
        materials = []
        patterns = []

        for pattern in self.patterns:
            self.generate_sample(pattern)
            materials.extend(self.materials)
            patterns.extend([pattern]*self.total_units)

        # Now we have materials and patterns lists of length 4000
        # Compute moving averages over a window
        window_size = 10  # Adjust window size as needed
        a_values = np.convolve([1 if m == 'A' else 0 for m in materials],
                               np.ones(window_size)/window_size, mode='valid')
        b_values = np.convolve([1 if m == 'B' else 0 for m in materials],
                               np.ones(window_size)/window_size, mode='valid')
        c_values = np.convolve([1 if m == 'C' else 0 for m in materials],
                               np.ones(window_size)/window_size, mode='valid')
        d_values = np.convolve([1 if m == 'D' else 0 for m in materials],
                               np.ones(window_size)/window_size, mode='valid')
        e_values = np.convolve([1 if m == 'E' else 0 for m in materials],
                               np.ones(window_size)/window_size, mode='valid')
        
        # Adjust patterns list to match the length of the moving averages
        patterns = patterns[window_size-1:]
        self.plot_inputs(a_values, b_values, c_values, d_values, e_values,
                         patterns, 'Seasonal Input Patterns (Ordered Patterns 1-4)')

def test_seed_reproducibility(seed, batchsize):
    # Initialize the generator with a seed
    generator1 = SeasonalInputGenerator(seed=seed)
    batch1 = generator1.generate_input(batchsize=batchsize)

    # Initialize another generator with the same seed
    generator2 = SeasonalInputGenerator(seed=seed)
    batch2 = generator2.generate_input(batchsize=batchsize)

    # Check if both batches are the same
    are_batches_equal = np.array_equal(batch1, batch2)
    print(f"Are both batches equal? {are_batches_equal}")  # Should output: True

    return are_batches_equal

# ---------------------------------------------------------*/
# Main
# ---------------------------------------------------------*/
if __name__ == "__main__":
    # Create an instance of the generator with a specific seed
    generator = SeasonalInputGenerator(seed=42)

    # Generate a batch of inputs
    batch = generator.generate_input(batchsize=100)
    print(batch)

    # Plot the seasonal input patterns over time
    generator.plot_seasonal_inputs()

    # Plot ordered patterns
    generator.plot_ordered_patterns()
    
    test_seed_reproducibility(seed=42, batchsize=100)

# -------------------------Notes-----------------------------------------------*\
# This class generates seasonal input patterns for a sorting system with five materials (A, B, C, D, E).
# The patterns are generated based on the following rules:
# - Pattern 1: More A+C (both 30-35%) than others
# - Pattern 2: More B+C (both 30-35%) than others
# - Pattern 3: More D+B (both 30-35%) than others
# -----------------------------------------------------------------------------*\
