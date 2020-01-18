from google.cloud import vision
import io

def detect_labels(path):
    """Detects labels in the file."""
    
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    print('Labels:')
    tlabels = []
    for label in labels:
        tlabels.append((label.score, label.description))
    
    response = client.web_detection(image=image)
    annotations = response.web_detection

    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            print(label)
            print('\nBest guess label: {}'.format(label.label))

    ent = []
    if annotations.web_entities:
        print('\n{} Web entities found: '.format(
            len(annotations.web_entities)))
        
        ent = []
        for entity in annotations.web_entities:
            if (entity.description != ""):
                ent.append((entity.score, entity.description))
            # print('\n\tScore      : {}'.format(entity.score))
            # print(u'\tDescription: {}'.format(entity.description))

    return tlabels, ent

def analyzer(labels, ent):
    detect = [
        'toddler',
        'sharp',
        'shoes',
        'dirt'
    ]

    similarWords = [
        ["child", "kid", "infant", "youngster", "preschooler"],
        ["razor", "knife", "blade", "violent","cutting"],
        ["boots", ""]
    ]
    
labels, entities = detect_labels("/Users/master/Desktop/rob3.jpg")
print("---------")
print(labels)
print("---------")
print(entities)
# detect_web("/Users/master/Desktop/rob3.jpg")