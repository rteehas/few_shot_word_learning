import json

def process_json_data(file_path):
    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    C_correct = 0
    D_correct = 0
    C_approx_correct = 0
    D_approx_correct = 0
    # Initialize counters
    both_original_true_count = 0
    both_approx_true_count = 0
    list_incorrect = []
    total = 0
    # Loop through each key and value in the 'generated_text' dictionary
    for key, value in data.items():
        print(f"Key: {key}")
        C = value['generated_text']['C']
        D = value['generated_text']['D']
        C_approx = value['generated_text']['C_approx = A - B + D']
        D_approx = value['generated_text']['D_approx = B - A + D']

        total += 1

        is_C_male = C.startswith("A: Male")
        is_C_female = C.startswith("A: Female")
        is_D_female = D.startswith("A: Female")
        is_D_male = D.startswith("A: Male")
        is_C_approx_male = C_approx.startswith("A: Male")
        is_C_approx_female = C_approx.startswith("A: Female")
        is_D_approx_female = D_approx.startswith("A: Female")
        is_D_approx_male = D_approx.startswith("A: Male")

        if is_C_male and is_D_female:
            both_original_true_count += 1  # Increment when both original conditions are true
            if is_C_approx_male and is_D_approx_female:
                both_approx_true_count += 1  # Increment when both approx conditions are true under the original condition
            else:
                list_incorrect.append(key)

        if is_C_male:
            C_correct += 1
        elif is_C_female:
            print("C Female: ", C.split('\n', 1)[0])
        else:
            print("unexpected C: ", C.split('\n', 1)[0])

        if is_D_female:
            D_correct += 1
        elif is_D_male:
            print("D Male: ", D.split('\n', 1)[0])
        else:
            print("unexpected D: ", D.split('\n', 1)[0])

        if is_C_approx_male:
            C_approx_correct += 1
        elif is_C_approx_female:
            print("C Female: ", C_approx.split('\n', 1)[0])
        else:
            print("unexpected C: ", C_approx.split('\n', 1)[0])

        if is_D_approx_female:
            D_approx_correct += 1
        elif is_D_approx_male:
            print("D Male: ", D_approx.split('\n', 1)[0])
        else:
            print("unexpected D: ", D_approx.split('\n', 1)[0])
        
    
        
    print(f"Total: {total}")
    print(f"C correct: {C_correct}")
    print(f"D correct: {D_correct}")
    print(f"C approx correct: {C_approx_correct}")
    print(f"D approx correct: {D_approx_correct}")
    # print the percentages
    print(f"C correct percentage: {C_correct/total}")
    print(f"D correct percentage: {D_correct/total}")
    print(f"C approx correct percentage: {C_approx_correct/total}")
    print(f"D approx correct percentage: {D_approx_correct/total}")

    # After processing, you can print or use the counters as needed
    print("Both original conditions true count:", both_original_true_count)
    print("Both approx conditions true under original condition count:", both_approx_true_count)
    print("Incorrect keys:", list_incorrect)

# main function that uses the process_json_data function
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python tmp.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    process_json_data(file_path)


