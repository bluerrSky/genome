import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional, Union, Any

import click
from midiutil import MIDIFile
from pyo import Server, Events, EventScale, EventSeq, Metro, CosTable, TrigEnv, Iter, Sine

Genome = List[int]
FitnessFunction = Callable[[Genome], int]

BITS_PER_NOTE = 4
KEYS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
SCALES = ["major", "minorM", "dorian", "phrygian", "lydian", "mixolydian", "majorBlues", "minorBlues"]

def int_from_bits(bits: List[int]) -> int:
    return sum(bit * (2 ** index) for index, bit in enumerate(bits))

def generate_genome(size: int) -> Genome:
    return [random.randint(0, 1) for _ in range(size)]

def selection_pair(population: List[Genome], fitness_func: FitnessFunction) -> Tuple[Genome, Genome]:
    return random.choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )

def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes must be of same length")

    length = len(a)
    if length < 2:
        return a, b

    p = random.randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = random.randrange(len(genome))
        if random.random() < probability:
            genome[index] = 1 if genome[index] == 0 else 0
    return genome

class MelodyGenerator:
    def __init__(
        self,
        num_bars: int,
        num_notes: int,
        num_steps: int,
        pauses: bool,
        key: str,
        scale: str,
        root: int,
        bpm: int
    ):
        self.num_bars = num_bars
        self.num_notes = num_notes
        self.num_steps = num_steps
        self.pauses = pauses
        self.key = key
        self.scale = scale
        self.root = root
        self.bpm = bpm
        self.server: Optional[Server] = None

    def genome_to_melody(self, genome: Genome) -> Dict[str, list]:
        notes = [
            genome[i * BITS_PER_NOTE:(i + 1) * BITS_PER_NOTE]
            for i in range(self.num_bars * self.num_notes)
        ]

        note_length = 4 / float(self.num_notes)
        scl = EventScale(root=self.key, scale=self.scale, first=self.root)

        melody = {
            "notes": [],
            "velocity": [],
            "beat": []
        }

        for note in notes:
            integer = int_from_bits(note)

            if not self.pauses:
                integer = integer % (2 ** (BITS_PER_NOTE - 1))

            if integer >= (2 ** (BITS_PER_NOTE - 1)):
                melody["notes"].append(0)
                melody["velocity"].append(0)
                melody["beat"].append(note_length)
            else:
                if melody["notes"] and melody["notes"][-1] == integer:
                    melody["beat"][-1] += note_length
                else:
                    melody["notes"].append(integer)
                    melody["velocity"].append(127)
                    melody["beat"].append(note_length)

        steps = []
        for step in range(self.num_steps):
            steps.append([scl[(note + step * 2) % len(scl)] for note in melody["notes"]])

        melody["notes"] = steps
        return melody

    def genome_to_events(self, genome: Genome) -> List[Events]:
        melody = self.genome_to_melody(genome)

        return [
            Events(
                midinote=EventSeq(step, occurrences=1),
                midivel=EventSeq(melody["velocity"], occurrences=1),
                beat=EventSeq(melody["beat"], occurrences=1),
                attack=0.001,
                decay=0.05,
                sustain=0.5,
                release=0.005,
                bpm=self.bpm
            ) for step in melody["notes"]
        ]

    def create_metronome(self) -> Any:
        met = Metro(time=60 / self.bpm).play()
        t = CosTable([(0, 0), (50, 1), (200, .3), (500, 0)])
        amp = TrigEnv(met, table=t, dur=.25, mul=1)
        freq = Iter(met, choice=[660, 440, 440, 440])
        return Sine(freq=freq, mul=amp).mix(2).out()

    def play_genome(self, genome: Genome) -> List[Events]:
        if not self.server:
            self.init_server()
            
        events = self.genome_to_events(genome)
        for event in events:
            event.play()
        self.server.start()
        return events

    def stop_playback(self, events: List[Events]) -> None:
        if self.server:
            self.server.stop()
            for event in events:
                event.stop()
            time.sleep(0.5)

    def init_server(self) -> None:
        self.server = Server().boot()
        time.sleep(0.5)

    def evaluate_fitness(self, genome: Genome) -> int:
        metronome = self.create_metronome()
        events = self.play_genome(genome)
        
        rating = input("Rating (0-5): ")
        
        self.stop_playback(events)
        
        try:
            return int(rating)
        except ValueError:
            return 0

    def save_to_midi(self, genome: Genome, filename: Union[str, Path]) -> None:
        melody = self.genome_to_melody(genome)
        
        if (len(melody["notes"][0]) != len(melody["beat"]) or 
            len(melody["notes"][0]) != len(melody["velocity"])):
            raise ValueError("Melody components have inconsistent lengths")

        midi = MIDIFile(1)
        track = 0
        channel = 0
        time_pos = 0.0
        
        midi.addTrackName(track, time_pos, "Genetic Algorithm Melody")
        midi.addTempo(track, time_pos, self.bpm)

        for i, velocity in enumerate(melody["velocity"]):
            if velocity > 0:
                for step in melody["notes"]:
                    midi.addNote(
                        track=track, 
                        channel=channel,
                        pitch=step[i],
                        time=time_pos,
                        duration=melody["beat"][i],
                        volume=velocity
                    )
            time_pos += melody["beat"][i]

        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "wb") as f:
            midi.writeFile(f)

class GeneticMusicEvolver:
    def __init__(
        self,
        melody_generator: MelodyGenerator,
        population_size: int = 10,
        num_mutations: int = 2,
        mutation_probability: float = 0.5
    ):
        self.melody_generator = melody_generator
        self.population_size = population_size
        self.num_mutations = num_mutations
        self.mutation_probability = mutation_probability
        self.genome_size = (
            melody_generator.num_bars * 
            melody_generator.num_notes * 
            BITS_PER_NOTE
        )
        
        self.population = [
            generate_genome(self.genome_size) 
            for _ in range(population_size)
        ]
        self.population_id = 0
        self.output_dir = Path(str(int(datetime.now().timestamp())))

    def evolve_generation(self) -> List[Tuple[Genome, int]]:
        random.shuffle(self.population)
        
        population_fitness = [
            (genome, self.melody_generator.evaluate_fitness(genome)) 
            for genome in self.population
        ]
        
        sorted_population = sorted(
            population_fitness, 
            key=lambda x: x[1], 
            reverse=True
        )
        
        self.population = [genome for genome, _ in sorted_population]
        
        next_generation = self.population[:2]
        
        for _ in range(self.population_size // 2 - 1):
            def fitness_lookup(genome: Genome) -> int:
                for g, fitness in population_fitness:
                    if g == genome:
                        return fitness
                return 0
            
            parents = selection_pair(self.population, fitness_lookup)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutation(
                offspring_a, 
                num=self.num_mutations, 
                probability=self.mutation_probability
            )
            offspring_b = mutation(
                offspring_b, 
                num=self.num_mutations, 
                probability=self.mutation_probability
            )
            
            next_generation.extend([offspring_a, offspring_b])
        
        print(f"Population {self.population_id} evaluation complete")
        self.population_id += 1
        self.population = next_generation
        
        return sorted_population
    
    def save_population(self) -> None:
        print("Saving population as MIDI files...")
        
        for i, genome in enumerate(self.population):
            output_path = (
                self.output_dir / 
                str(self.population_id - 1) / 
                f"{self.melody_generator.scale}-{self.melody_generator.key}-{i}.mid"
            )
            self.melody_generator.save_to_midi(genome, output_path)
        
        print("Done saving files.")
    
    def demo_top_genomes(self) -> None:
        if len(self.population) < 2:
            print("Not enough genomes to demo")
            return
        
        print("Playing the top-rated melody...")
        events = self.melody_generator.play_genome(self.population[0])
        input("Press Enter to stop...")
        self.melody_generator.stop_playback(events)
        
        print("Playing the second-best melody...")
        events = self.melody_generator.play_genome(self.population[1])
        input("Press Enter to stop...")
        self.melody_generator.stop_playback(events)

@click.command()
@click.option("--num-bars", default=8, prompt="Number of bars", type=int)
@click.option("--num-notes", default=4, prompt="Notes per bar", type=int)
@click.option("--num-steps", default=1, prompt="Number of steps", type=int)
@click.option("--pauses", default=True, prompt="Introduce pauses", type=bool)
@click.option("--key", default="C", prompt="Key", type=click.Choice(KEYS, case_sensitive=False))
@click.option("--scale", default="major", prompt="Scale", type=click.Choice(SCALES, case_sensitive=False))
@click.option("--root", default=4, prompt="Scale root", type=int)
@click.option("--population-size", default=10, prompt="Population size", type=int)
@click.option("--num-mutations", default=2, prompt="Number of mutations", type=int)
@click.option("--mutation-probability", default=0.5, prompt="Mutation probability", type=float)
@click.option("--bpm", default=128, prompt="Beats per minute", type=int)
def main(
    num_bars: int,
    num_notes: int,
    num_steps: int,
    pauses: bool,
    key: str,
    scale: str,
    root: int,
    population_size: int,
    num_mutations: int,
    mutation_probability: float,
    bpm: int
) -> None:
    melody_generator = MelodyGenerator(
        num_bars=num_bars,
        num_notes=num_notes,
        num_steps=num_steps,
        pauses=pauses,
        key=key,
        scale=scale,
        root=root,
        bpm=bpm
    )
    
    melody_generator.init_server()
    
    evolver = GeneticMusicEvolver(
        melody_generator=melody_generator,
        population_size=population_size,
        num_mutations=num_mutations,
        mutation_probability=mutation_probability
    )
    
    running = True
    while running:
        evolver.evolve_generation()
        evolver.demo_top_genomes()
        evolver.save_population()
        
        running = input("Continue to next generation? [Y/n] ").lower() != "n"

if __name__ == "__main__":
    main()
