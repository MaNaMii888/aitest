import numpy as np
import matplotlib.pyplot as plt
import random

# ฟังก์ชันเป้าหมาย
def objective_function(x):
    return x * np.sin(10 * np.pi * x) + 2.0

# สร้างประชากรเริ่มต้น
def create_initial_population(pop_size, chromosome_length):
    population = []
    # สุ่มโครโมโซมในช่วง 0 ถึง 1 โดยใช้การสุ่มเลขฐานสองเพื่อสร้างโครโมโซม
    for _ in range(pop_size):
        chromosome = [random.randint(0, 1) for _ in range(chromosome_length)] # สุ่มเลขฐานสอง 0 หรือ 1
        population.append(chromosome) # สร้างโครโมโซมใหม่
    return population

# แปลงโครโมโซมเป็นค่า x
def decode_chromosome(chromosome, min_value, max_value):
    # เติมโค้ดที่นี่: แปลงโครโมโซมแบบไบนารีเป็นค่าจริงในช่วง [min_value, max_value]
    chromosome_length = len(chromosome) # ความยาวของโครโมโซม ,แปลงโครโมโซมเป็นค่า x โดยใช้การแปลงเลขฐานสอง
    # แปลงโครโมโซมเป็นเลขฐานสิบ
    chromosome_as_int = sum([bit * (2 ** i) for i, bit in enumerate(reversed(chromosome))]) # แปลงเป็นเลขฐานสิบ #เป็นวิธีในการสร้าง list #bit = 1 หรือ 0 (2**i คือสองที่ยกกำลัง i )
    # คำนวณค่า x โดยใช้การแปลงจากเลขฐานสองเป็นค่าจริง
    x = min_value + (max_value - min_value) * (chromosome_as_int / (2 ** chromosome_length - 1))
    #max_value - min_value คือช่วงของค่าที่เราต้องการแปลง
    # (2 ** chromosome_length - 1) คือค่าที่มากที่สุดที่สามารถแสดงได้ด้วยโครโมโซมนี้ -1 เพราะเริ่มนับจาก 0
    return x

# คำนวณความเหมาะสม
def calculate_fitness(population, min_value, max_value): #population คือประชากรที่เราสุ่มขึ้นมา min_value และ max_value คือค่าต่ำสุดและสูงสุดที่เราต้องการ
    # เติมโค้ดที่นี่: คำนวณค่าความเหมาะสมของแต่ละโครโมโซมในประชากร
    fitness_values = [] # สร้าง list สำหรับเก็บค่าความเหมาะสม
    for chromosome in population: #เพื่อวนลูปในประชากร
        x = decode_chromosome(chromosome, min_value, max_value) # แปลงโครโมโซมเป็นค่า x
        fitness = objective_function(x) # คำนวณค่าฟังก์ชันเป้าหมาย
        fitness_values.append(fitness) # เก็บค่าความเหมาะสมใน list
    return fitness_values

# การคัดเลือกแบบ Roulette Wheel Selection พร้อม Elitism
def selection(population, fitness_values, num_elites=2):
    # เติมโค้ดที่นี่: เลือกโครโมโซมโดยใช้วิธี Roulette Wheel Selection
    pop_size = len(population) # จำนวนประชากร
    total_fitness = sum(fitness_values) # คำนวณค่าความเหมาะสมรวม
    if total_fitness == 0: # ถ้าค่าความเหมาะสมรวมเป็น 0 ให้ใช้การสุ่ม
        selection_probs = [1 / pop_size] * pop_size
    else:  # คำนวณความน่าจะเป็นในการเลือกโครโมโซม
        selection_probs = [f / total_fitness for f in fitness_values]
    cumulative_probabilities = np.cumsum(selection_probs)
    selected_population = [] # สร้าง list สำหรับเก็บประชากรที่ถูกเลือก

    # คัดเลือกประชากรโดยใช้ Elitism
    # Elitism: เก็บโครโมโซมที่ดีที่สุดไว้
    sorted_indices = np.argsort(fitness_values)[::-1] # เรียงลำดับค่าความเหมาะสมจากมากไปน้อย
    elites = [population[i] for i in sorted_indices[:num_elites]] # เก็บโครโมโซมที่ดีที่สุด
    selected_population.extend(elites) # เพิ่มโครโมโซมที่ดีที่สุดในประชากรที่ถูกเลือก

    # Roulette Wheel Selection สำหรับส่วนที่เหลือ
    for _ in range(pop_size - num_elites): # จำนวนประชากรที่เหลือหลังจากเลือกโครโมโซมที่ดีที่สุด
        # สุ่มค่าเพื่อเลือกโครโมโซมจากประชากร
        rand = random.random()
        for i, prob in enumerate(cumulative_probabilities):
            if rand <= prob: # ถ้าค่าที่สุ่มอยู่ในช่วงความน่าจะเป็นที่คำนวณได้
                selected_population.append(population[i]) # เพิ่มโครโมโซมที่ถูกเลือกในประชากรที่ถูกเลือก
                break
    return selected_population

# การผสมพันธุ์
def crossover(parent1, parent2, crossover_rate):
    # เติมโค้ดที่นี่: ทำการผสมพันธุ์ระหว่างพ่อและแม่
    if random.random() < crossover_rate: # ถ้าค่าที่สุ่มน้อยกว่าค่าการผสมพันธุ์
        crossover_point = random.randint(1, len(parent1) - 1) # สุ่มจุดตัดการผสมพันธุ์
        # สร้างลูกโครโมโซมโดยการผสมพันธุ์ระหว่างพ่อแม่
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return list(child1), list(child2)
    else: # ถ้าไม่ทำการผสมพันธุ์ ให้ส่งพ่อแม่กลับไปโดยไม่เปลี่ยนแปลงค่า
        return list(parent1), list(parent2)

# การกลายพันธุ์
def mutation(chromosome, mutation_rate):
    # เติมโค้ดที่นี่: ดำเนินการตามขั้นตอนของ GA
    mutated_chromosome = list(chromosome) # สร้างสำเนาของโครโมโซมเพื่อทำการกลายพันธุ์
    # ทำการกลายพันธุ์โดยการสุ่มเปลี่ยนค่า 0 เป็น 1 หรือ 1 เป็น 0
    for i in range(len(mutated_chromosome)): # วนลูปในแต่ละบิตของโครโมโซม
        if random.random() < mutation_rate: # ถ้าค่าที่สุ่มน้อยกว่าค่าการกลายพันธุ์
            mutated_chromosome[i] = 1 - mutated_chromosome[i] # เปลี่ยนค่า 0 เป็น 1 หรือ 1 เป็น 0
    #mutation_rate คืออัตราการกลายพันธุ์ที่เราต้องการให้เกิดขึ้นในโครโมโซม
    return mutated_chromosome

# อัลกอริทึมพันธุกรรม
def genetic_algorithm(pop_size=100, chromosome_length=20, min_value=-1.0, max_value=2.0, generations=100, crossover_rate=0.7, mutation_rate=0.01, fitness_threshold=10.0, convergence_threshold=20):
    #pop_size คือจำนวนประชากรที่เราต้องการสร้างขึ้นมา chromosome_length คือความยาวของโครโมโซม min_value และ max_value คือค่าต่ำสุดและสูงสุดที่เราต้องการให้โครโมโซมแสดงผล generations คือจำนวนรุ่นที่เราต้องการให้ทำงาน crossover_rate คืออัตราการผสมพันธุ์ mutation_rate คืออัตราการกลายพันธุ์
    # สร้างประชากรเริ่มต้น
    population = create_initial_population(pop_size, chromosome_length)
    all_best_fitness = []  # สร้าง list สำหรับเก็บค่าฟิตเนสที่ดีที่สุดในแต่ละรุ่น
    # ค่าฟิตเนสที่ดีที่สุดในรุ่นแรก
    best_solution = None
    best_fitness = float('-inf') # ค่าฟิตเนสที่ดีที่สุดเริ่มต้นเป็นค่าต่ำสุด
    convergence_count = 0
    previous_best_fitness = float('-inf')

    # วนลูปตามจำนวนรุ่นที่กำหนด
    for generation in range(generations):
        fitness_values = calculate_fitness(population, min_value, max_value) # คำนวณค่าฟิตเนสของประชากรในรุ่นนี้
        best_fitness_current_gen = max(fitness_values) # ค่าฟิตเนสที่ดีที่สุดในรุ่นนี้
        best_index_current_gen = fitness_values.index(best_fitness_current_gen) # ค้นหา index ของโครโมโซมที่ดีที่สุดในรุ่นนี้
        best_chromosome_current_gen = population[best_index_current_gen]   # ค้นหาโครโมโซมที่ดีที่สุดในรุ่นนี้
        best_solution_current_gen = decode_chromosome(best_chromosome_current_gen, min_value, max_value) # แปลงโครโมโซมที่ดีที่สุดเป็นค่า x
        # อัปเดตค่าฟิตเนสที่ดีที่สุดและโครโมโซมที่ดีที่สุดถ้าค่าฟิตเนสในรุ่นนี้ดีกว่าค่าฟิตเนสที่ดีที่สุดก่อนหน้านี้

        if best_fitness_current_gen > best_fitness: # ถ้าค่าฟิตเนสในรุ่นนี้ดีกว่าค่าฟิตเนสที่ดีที่สุดก่อนหน้านี้
            best_fitness = best_fitness_current_gen # อัปเดตค่าฟิตเนสที่ดีที่สุด
            best_solution = best_solution_current_gen   # อัปเดตโครโมโซมที่ดีที่สุด
            convergence_count = 0  # Reset convergence count
        else:
            convergence_count += 1

        all_best_fitness.append(best_fitness) # เก็บค่าฟิตเนสที่ดีที่สุดในรุ่นนี้ใน list
        # แสดงผลลัพธ์ในแต่ละรุ่น
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness_current_gen:.4f}, Best Solution = {best_solution_current_gen:.4f}")
        # คัดเลือกประชากรเพื่อสร้างรุ่นถัดไป

        selected_population = selection(population, fitness_values) # คัดเลือกประชากรโดยใช้ Elitism
        # สร้างประชากรใหม่โดยใช้การผสมพันธุ์และการกลายพันธุ์
        new_population = [] # สร้าง list สำหรับเก็บประชากรใหม่
        for i in range(0, pop_size, 2): # วนลูปในประชากรที่ถูกเลือกเป็นคู่
            # สุ่มเลือกพ่อแม่เพื่อทำการผสมพันธุ์
            parent1 = selected_population[i % pop_size]
            parent2 = selected_population[(i + 1) % pop_size]
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            new_population.append(mutation(child1, mutation_rate)) # ทำการกลายพันธุ์ในลูกโครโมโซม
            if len(new_population) < pop_size: # ถ้าจำนวนประชากรใหม่ยังไม่ถึงจำนวนที่ต้องการ
                # ทำการกลายพันธุ์ในลูกโครโมโซม
                new_population.append(mutation(child2, mutation_rate))
        population = new_population # อัปเดตประชากรเป็นประชากรใหม่

        # เงื่อนไขการหยุดทำงาน
        if best_fitness >= fitness_threshold:
            print(f"GA stopped: Fitness threshold reached at generation {generation + 1}")
            break
        if convergence_count >= convergence_threshold:
            print(f"GA stopped: Convergence reached at generation {generation + 1}")
            break

    # คำนวณค่าฟิตเนสของประชากรสุดท้าย
    return best_solution, best_fitness, all_best_fitness

# ทดสอบอัลกอริทึม
if __name__ == "__main__":
    best_solution, best_fitness, all_best_fitness = genetic_algorithm()

    # แสดงผลลัพธ์
    print(f"\nค่า x ที่ดีที่สุดที่พบ: {best_solution:.4f}")
    print(f"ค่าฟังก์ชันสูงสุดที่พบ: {best_fitness:.4f}")

    # แสดงกราฟ
    plt.figure(figsize=(12, 6))

    # กราฟแสดงค่าฟังก์ชัน
    plt.subplot(1, 2, 1)
    x = np.linspace(-1, 2, 1000)
    y = objective_function(x)
    plt.plot(x, y)
    plt.scatter(best_solution, best_fitness, color='red', marker='o', s=100)
    plt.title('Objective Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')

    # กราฟแสดงการลู่เข้า
    plt.subplot(1, 2, 2)
    plt.plot(all_best_fitness)
    plt.title('Convergence')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')

    plt.tight_layout()
    plt.show()