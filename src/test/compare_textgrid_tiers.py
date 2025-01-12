import os
import csv
from textgrid import TextGrid
label_dict = {}
def load_textgrid(file_path):
    tg = TextGrid()
    tg.read(file_path)
    return tg

def get_tier_marks(tier):
    return [point.mark for point in tier.points]

def calculate_accuracy(marks1, marks2):
    if not marks1 and not marks2:  # 빈 경우 계산 안함
        return 1.0

    if len(marks1) != len(marks2):  # 개수가 다르면 틀린 것으로 적용
        return 0.0
    
    # 전체 순서에 맞게 맞힌 개수를 계산한다.
    matching_count = sum(1 for m1, m2 in zip(marks1, marks2) if m1 == m2)
    return matching_count / len(marks1) if len(marks1) > 0 else 0.0

def normalize_marks(marks):
    return [mark.split('(')[0] for mark in marks]

def calculate_agreement(original_path, new_path):
    results = []
    unique_marks = set()
    original_files = [f for f in os.listdir(original_path) if f.endswith('.TextGrid')]
    new_files = [f for f in os.listdir(new_path) if f.endswith('.TextGrid')]

    common_files = set(original_files) & set(new_files)

    for file_name in common_files:
        original_file = os.path.join(original_path, file_name)
        new_file = os.path.join(new_path, file_name)

        original_tg = load_textgrid(original_file)
        new_tg = load_textgrid(new_file)

        original_ap_tier = original_tg.getFirst("AP")
        original_ap_medial_tier = original_tg.getFirst("AP-medial")

        new_ap_tier = new_tg.getFirst("AP")
        new_ap_medial_tier = new_tg.getFirst("AP-medial")

        original_ap_marks = get_tier_marks(original_ap_tier)
        original_ap_medial_marks = get_tier_marks(original_ap_medial_tier)

        # Normalize marks to handle H(L) -> H transformation
        new_ap_marks = normalize_marks(get_tier_marks(new_ap_tier))
        new_ap_medial_marks = normalize_marks(get_tier_marks(new_ap_medial_tier))

        # Add marks to the unique set
        unique_marks.update(original_ap_marks)
        unique_marks.update(original_ap_medial_marks)
        unique_marks.update(new_ap_marks)
        unique_marks.update(new_ap_medial_marks)
        for ori_ap in original_ap_marks:
            if ori_ap in label_dict:
                label_dict[ori_ap] += 1
            elif ori_ap not in label_dict:
                label_dict[ori_ap] = 1
        for new_ap in new_ap_marks:
            if new_ap in label_dict:
                label_dict[new_ap] += 1
            elif new_ap not in label_dict:
                label_dict[new_ap] = 1
        for ori_ap_m in original_ap_medial_marks:
            if ori_ap_m in label_dict:
                label_dict[ori_ap_m] += 1
            elif ori_ap_m not in label_dict:
                label_dict[ori_ap_m] = 1
        for new_ap_m in new_ap_medial_marks:
            if new_ap_m in label_dict:
                label_dict[new_ap_m] += 1
            elif new_ap_m not in label_dict:
                label_dict[new_ap_m] = 1
        ap_accuracy = calculate_accuracy(original_ap_marks, new_ap_marks)
        ap_medial_accuracy = calculate_accuracy(original_ap_medial_marks, new_ap_medial_marks)

        results.append((file_name, ap_accuracy, ap_medial_accuracy))

    # Print unique marks
    print(f"Unique marks: {unique_marks}")
    print(f"Total unique marks: {len(unique_marks)}")
    print(label_dict)

    return results

def save_results_to_csv(results, output_file='out/test/comparison_results.csv'):
    if not os.path.exists('out/test'):
        os.makedirs('out/test')
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["File Name", "AP Tier Accuracy", "AP-medial Tier Accuracy"])
        for result in results:
            writer.writerow(result)

def calculate_average_agreement(results):
    ap_agreements = [result[1] for result in results]
    ap_medial_agreements = [result[2] for result in results]

    avg_ap_agreement = sum(ap_agreements) / len(ap_agreements) if ap_agreements else 0
    avg_ap_medial_agreement = sum(ap_medial_agreements) / len(ap_medial_agreements) if ap_medial_agreements else 0

    return avg_ap_agreement, avg_ap_medial_agreement

if __name__ == "__main__":
    original_path = "data/sample/"
    new_path = "out/textgrid-v0.1.1"
    output_csv = "out/test/comparison_results.csv"

    agreements = calculate_agreement(original_path, new_path)
    save_results_to_csv(agreements, output_csv)

    avg_ap_agreement, avg_ap_medial_agreement = calculate_average_agreement(agreements)

    print(f"Average AP Tier Accuracy: {avg_ap_agreement:.2f}")
    print(f"Average AP-medial Tier Accuracy: {avg_ap_medial_agreement:.2f}")
