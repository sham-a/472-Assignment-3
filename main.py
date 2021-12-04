import gensim.downloader as g
import csv


def task1():
    model = g.load('word2vec-google-news-300')

    f = open("synonyms.txt", "r")
    writer = open('word2vec-google-news-300-details.csv', 'w')
    csv_write = csv.writer(writer)
    csv_write.writerow(['Question Word', 'Correct Word', 'Guessed Word', 'Label'])

    indexes = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3
    }

    correct_count = 0
    guess_count = 0

    while True:
        line_reader = f.readline()
        if not line_reader:
            break
        row = []
        question_word = line_reader.split('	')[1].strip('\n')
        row.append(question_word)
        guess_words = []
        cosines = []
        in_model = True

        if question_word not in model.key_to_index:
            in_model = False

        for i in range(4):
            line = f.readline()
            guess = line.split('	')[1].strip('\n')
            guess_words.append(guess)
            if guess in model.key_to_index and in_model:
                cosine = model.similarity(question_word, guess)
                cosines.append(cosine)
            else:
                cosines.append(-2)

        answer_letter = f.readline()[0]
        answer = guess_words[indexes[answer_letter]]

        index_max = cosines.index(max(cosines))
        final_guess_word = guess_words[index_max]

        count = 0
        for cos in cosines:
            if cos == -2:
                count += 1

        if count == 4 or not in_model:
            label = 'guess'
            guess_count += 1
        elif final_guess_word == answer:
            label = 'correct'
            correct_count += 1
        else:
            label = 'wrong'

        row.append(answer)
        row.append(final_guess_word)
        row.append(label)
        csv_write.writerow(row)

    f.close()
    writer.close()

    # part 2
    writer_2 = open('analysis.csv', 'w')
    csv_2 = csv.writer(writer_2)
    csv_2.writerow(['Model', 'Size', 'C', 'V', 'Accuracy'])
    row_2 = ['word2vec-google-news-300']

    size = len(model)
    row_2.append(size)
    row_2.append(correct_count)
    v = 80 - guess_count
    row_2.append(v)
    accuracy = correct_count/v
    row_2.append(accuracy)

    csv_2.writerow(row_2)

    writer_2.close()


if __name__ == '__main__':
    task1()
