#Global
import numpy as np

classes = ["car", "bike", "truck", "motorcycles"]

#%% FILES
import os
from skimage.io import imread
from skimage.transform import resize


def file_loader(path):
    images, labels, file_names = [],[],[]

    print(f"Cargando DataSet desde: {path}\n")

    for clase in classes:

        class_path = os.path.join(path,clase)
        files = os.listdir(class_path)

        print(f"{clase}: {len(files)} imágenes\n")

        for file in files:
            
            file_path = os.path.join(class_path,file)
            
            try:
                img = imread(file_path, as_gray = True)
                img = resize(img, (512,512), anti_aliasing=True)
                images.append(img)
                labels.append(clase)
                file_names.append(os.path.basename(file_path))
            
            except Exception as e:
                print(f"Error cargando {file_path}: {e}\n")
    
    return {'images': images, 'labels': labels, 'file_names': file_names }

#%% CROSS VALIDATION SETS


def idx_lab_in_list(labels, clase):
  return [j for j, etiq in enumerate(labels) if etiq == clase]


def indir_idx(idx, idx_list, main_list):
  return main_list[idx_list[idx]]


def get_subset_portion(list, idx_class, subset_class_start, subset_class_end):
  return [indir_idx(idx, idx_class, list) for idx in range(int(subset_class_start), int(subset_class_end))]


def crear_subsets(images, labels, file_names, keypoints, descriptors, n_subsets):

    # Índices por categoría y cantidad por subset
    idx_labels = { cat: idx_lab_in_list(labels, cat) for cat in classes }

    n_per_subset = {cat: int(np.ceil(len(indices) / n_subsets)) for cat, indices in idx_labels.items()}

    # Mezclamos los índices de cada clase
    for idxs in idx_labels.values():
        np.random.shuffle(idxs)

    subsets = []

    for i in range(n_subsets):
        subset = {
            'images': [],
            'labels': [],
            'file_names': [],
            'keypoints': [],
            'descriptors': [],
            'stats': {clase: {'n_images': 0} for clase in classes}
        }

        for clase in classes:
            idxs = idx_labels[clase]
            ini = i * n_per_subset[clase]
            fin = min(ini + n_per_subset[clase], len(idxs))

            idx_slice = idxs[ini:fin]

            subset['images'] += [images[j] for j in idx_slice]
            subset['labels'] += [labels[j] for j in idx_slice]
            subset['file_names'] += [file_names[j] for j in idx_slice]
            subset['keypoints'] += [keypoints[j] for j in idx_slice]
            subset['descriptors'] += [descriptors[j] for j in idx_slice]
            subset['stats'][clase]['num_images'] = len(idx_slice)

        subsets.append(subset)

    return subsets

def subset_join(subsets):
    subset_union = {
        'images': [],
        'labels': [],
        'file_names': [],
        'keypoints': [],
        'descriptors': [],
        'stats': {clase: {'n_images': 0} for clase in classes}
    }

    for subset in subsets:
        subset_union['images'] += subset['images']
        subset_union['labels'] += subset['labels']
        subset_union['file_names'] += subset['file_names']
        subset_union['keypoints'] += subset['keypoints']
        subset_union['descriptors'] += subset['descriptors']

        for clase in classes:
            subset_union['stats'][clase]['n_images'] += subset['stats'][clase]['num_images']

    return subset_union

#%% FEATURE EXTRACTION

from skimage.feature import ORB

def extraerORBImagen(image,nkeypoints):

    orb = ORB(nkeypoints)
    orb.detect_and_extract(image)

    return orb.keypoints,orb.descriptors


def extraerORBDataSet(images, labels,nkeypoints):
  
  keypoints_dataset = []
  descriptors_dataset = []

  # Estadísticas por categoría
  stats = {clase: {'total_keypoints': 0, 'num_images': 0} for clase in classes}

  print("Extrayendo características ORB...")

  # i: indice | img: imagen correspondiente | etiqueta: etiqueta correspondiente
  for i, (img, label) in enumerate(zip(images, labels)):

    keypoints, descriptors = extraerORBImagen(img,nkeypoints)

    keypoints_dataset.append(keypoints)
    descriptors_dataset.append(descriptors)

    # Actualizar estadísticas
    stats[label]['total_keypoints'] += len(keypoints)
    stats[label]['num_images'] += 1

  # Calcular promedios
  for clase in stats:
    if stats[clase]['num_images'] > 0:
      stats[clase]['promedio_keypoints'] = stats[clase]['total_keypoints'] / stats[clase]['num_images']

  return keypoints_dataset, descriptors_dataset, stats

#%% BOW
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq

def build_descriptor_matrix(descriptor_list):

    descriptor_matrix = []

    for descriptors in descriptor_list:
        for descriptor in descriptors:
            descriptor_matrix.append(descriptor)

    return np.array(descriptor_matrix, dtype=np.float64)

def build_visual_vocabulary(descriptors, n_keypoints, n_visual_words):

    print(f"Construyendo vocabulario visual con {n_keypoints} keypoints y {n_visual_words} palabras visuales...")

    model = KMeans(n_visual_words)
    model.fit(descriptors)

    return model

def descriptors_to_bow(descriptors, vocabulary):

    n_clusters = vocabulary.n_clusters
    assignments, _ = vq(descriptors, vocabulary.cluster_centers_)
    histogram, _ = np.histogram(assignments, bins=n_clusters, 
                                range=(0, n_clusters - 1))
    
    return histogram.astype(float)

def build_bow_matrix(descriptor_list, vocabulary):

    print("Creando representacion BOW...")
    bow_matrix = []

    for descriptors in descriptor_list:
        bow_vector = descriptors_to_bow(descriptors, vocabulary)
        bow_matrix.append(bow_vector)

    return np.array(bow_matrix)

#%% CLASSIFICATION

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

def train_svm(train_descriptors, train_labels):

    clf = make_pipeline(SVC(kernel='linear', probability=True))
    clf.fit(train_descriptors, train_labels)

    return clf


def evaluate_svm_classifier(classifier, test_descriptors, test_labels, experiment_name=""):
 
    print(f"Evaluando Clasificador: {experiment_name}...")

    # Predict labels and class probabilities
    predictions = classifier.predict(test_descriptors)
    probabilities = classifier.predict_proba(test_descriptors)
    confidences = probabilities.max(axis=1)

    # Compute evaluation metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, predictions, average=None, labels=classes
    )

    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions, labels=classes)

    return {
        'predictions': predictions,
        'confidences': confidences,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm
    }

#%% CROSSVALIDATION

import time

def crossval(images, labels, file_names, keypoint_options, vocab_options,n_subsets=3):
    resultados = []

    for n_keypoints in keypoint_options:
        # ORB
        keypoints, descriptors, stats = extraerORBDataSet(images, labels, n_keypoints)

        # Particionamos en subsets balanceados
        subsets = crear_subsets(images, labels, file_names, keypoints, descriptors, n_subsets)

        for n_visual_words in vocab_options:
            print(f"\nEvaluando: {n_keypoints=}, {n_visual_words=}")

            accs = []
            f1s = []

            start_time = time.time()
            total_imgs = 0
            
            for i in range(n_subsets):
                # Subsets de train y test
                test_subset = subsets[i]
                train_subsets = [s for j, s in enumerate(subsets) if j != i]
                train_subset = subset_join(train_subsets)

                # BoW
                train_desc_matrix = build_descriptor_matrix(train_subset['descriptors'])
                vocab = build_visual_vocabulary(train_desc_matrix, n_keypoints, n_visual_words)

                X_train = build_bow_matrix(train_subset['descriptors'], vocab)
                y_train = train_subset['labels']

                X_test = build_bow_matrix(test_subset['descriptors'], vocab)
                y_test = test_subset['labels']

                # Clasificación
                clf = train_svm(X_train, y_train)
                resultados_test = evaluate_svm_classifier(clf, X_test, y_test)

                accs.append(resultados_test['accuracy'])
                f1s.append(np.mean(resultados_test['f1']))
                total_imgs += len(test_subset['images'])
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / total_imgs    
              
            # Guardar resultados promedio
            resultados.append({
              'n_keypoints': n_keypoints,
              'n_visual_words': n_visual_words,
              'accuracy_promedio': np.mean(accs),
              'f1_promedio': np.mean(f1s),
              'total_time': total_time,
              'avg_time': avg_time
            })

    return resultados

#%% ANALISIS
def best_result(results):
  sorted_results = sorted(results, key=lambda r: r['accuracy_promedio'], reverse=True)

  # Header
  print("="*108)
  print(f"{'n_keypoints':>12} | {'n_visual_words':>15} | {'sample_size':>12} | {'acc_prom':>10} | {'f1_prom':>8} | {'t_total(s)':>10} | {'t/img(s)':>8}")
  print("="*108)

  for res in sorted_results:
    print(f"{res['n_keypoints']:12} | {res['n_visual_words']:15} | {res['sample_size']:12} | " f"{res['accuracy_promedio']:.4f}   | {res['f1_promedio']:.4f}  | " f"{res['tiempo_total']:.2f}     | {res['tiempo_por_imagen']:.4f}")
    print("="*108)
  return max(sorted_results, key=lambda r: r['accuracy_promedio'])
  