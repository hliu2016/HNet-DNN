import os

# folder = './PREDICT'
folder = './CDataset'
case_disease = 'D114480'  # Breast cancer


def load_known_interaction(case_disease):
    files = ['train.txt', 'valid.txt', 'test.txt']

    known_drug = []
    for file in files:
        with open(os.path.join(folder, file), 'r') as f:
            for line in f:
                drug_id, disease_id, _ = line.strip().split()
                if disease_id == case_disease:
                    print('known: {} {}'.format(drug_id, disease_id))
                    known_drug.append(drug_id)
    return known_drug

if __name__ == '__main__':
    A = []  # disease - drug
    with open(os.path.join(folder, 'DiDrAMat'), 'r') as f:
        for line in f:
            adj = [int(x) for x in line.strip().split()]
            A.append(adj)

    disease_dict = dict()
    disease_list = []
    with open(os.path.join(folder, 'DiseasesName'), 'r') as f:
        for line in f:
            disease_dict[line.strip()] = len(disease_dict)
            disease_list.append(line.strip())

    drugDict = dict()
    drugList = []
    with open(os.path.join(folder, 'DrugsName'), 'r') as f:
        for line in f:
            drugDict[line.strip()] = len(drugDict)
            drugList.append(line.strip())

    disease_id = disease_dict[case_disease]
    print('case disease_id: {}'.format(disease_id))

    known_drugs = load_known_interaction(case_disease)

    select = [True for _ in range(len(drugList))]
    for drug in known_drugs:
        select[drugDict[drug]] = False

    with open(os.path.join(folder, 'case_breastCancer.txt'), 'w') as f:
        for i in range(len(select)):
            if select[i]:
                f.write('{} {}\n'.format(drugList[i], case_disease))

    print('done')

