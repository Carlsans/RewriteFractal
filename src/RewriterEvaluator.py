import glob
import gzip
import io
import math
import random
import statistics

import numpy as np
from difflib import SequenceMatcher
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from skimage import io, transform, color
from skimage.feature import hog  # Histogram of Oriented Gradients
#from src.RewriterEngine import RewriterEngine


class RewriterEvaluator:
    def __init__(self,conf):
        self.conf = conf
        self.HOG_set_features = None
    def setRewriter(self,rewriter):
        self.rewriter = rewriter

    def evaluateRewriter(self):
        linediff,vectorialvariablesdistribution,volumepenality = self.evaluateByLines()
        #print("volumepenality",volumepenality)
        #print("linediff,vectorialvariablesdistribution ",linediff,vectorialvariablesdistribution )
        #print("linediff,vectorialvariablesdistribution,volumepenality",linediff,vectorialvariablesdistribution,volumepenality)
        return linediff * (1 / volumepenality)

    def evaluateRewriterByCompressionRatio(self) -> float:
        """
        Compress a string using gzip and calculate the compression ratio.

        :param data: The input string to compress.
        :return: The ratio of compressed size to uncompressed size.
        """
        data = self.rewriter.getLinesAsString()
        if not data:
            return 0

        # Encode the string to bytes
        uncompressed_data = data.encode('utf-8')
        uncompressed_size = len(uncompressed_data)

        # Compress the data using gzip
        compressed_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed_buffer, mode='wb') as gzip_file:
            gzip_file.write(uncompressed_data)

        compressed_size = len(compressed_buffer.getvalue())

        # Calculate and return the compression ratio
        return compressed_size / uncompressed_size

    def volumepenality(self,line,uppercharacterpenality = 100):
        #print(line)
        #print(len(line))
        if len(line) <= uppercharacterpenality:

            return 1

        return math.pow((len(line)-uppercharacterpenality),2)
    def istriangle(self,depth = 20):
        lines = self.rewriter.stringlist
        lastlinelen = len(lines[0])
        for i in range(1, depth):
            if not (len(lines[i]) - lastlinelen) == 2:
                return False
            lastlinelen = len(lines[i])
        exit()
        return True
    def simpletrianglescore(self,depth = 100):
        lines = self.rewriter.stringlist
        score = 0
        lastlinelen = len(lines[0])
        for i in range(1, depth):
            score += abs(len(lines[i]) - lastlinelen - 2)
            lastlinelen = len(lines[i])
        return score/depth # Normalized
    def parabolictrianglescore(self,depth = 100):
        lines = self.rewriter.stringlist
        score = 0
        lastlinelen = len(lines[0])
        for i in range(1, depth):
            addedlines = len(lines[i]) - lastlinelen
            if addedlines <= 2:
                score += -addedlines*addedlines/4 +  addedlines
            else:
                score += -addedlines*addedlines/25 + (4/25)*addedlines + 21/25
            lastlinelen = len(lines[i])
        return score / depth # Normalized

    def trianglescore(self,depth = 20):
        lines = self.rewriter.stringlist
        #print("lines",lines)
        lastlinelen = len(lines[0])
        totaldifference = 0
        squaresum = 0
        #print(lines[0])
        for i in range(1,depth):
            #print(lines[i])
            totaldifference += len(lines[i]) - lastlinelen
            squaredifference = math.pow(2-(len(lines[i]) - lastlinelen),2)
            squaresum += squaredifference
            lastlinelen = len(lines[i])
        averagedifference = totaldifference / depth
        averagesquaresum = squaresum / depth
        # print("averagedifference",averagedifference)
        # print("averagesquaresum", averagesquaresum)
        # if averagesquaresum == 2 and averagedifference == 2:
        #     exit()
        differencenote = max(0,-(averagedifference*averagedifference)/4 + averagedifference)
        squaredifferencenote = max(0,-averagesquaresum*averagesquaresum + 1)
        note = (differencenote + squaredifferencenote)/2
        print("differencenote",differencenote)
        print("squaredifferencenote",squaredifferencenote)
        print("(differencenote + squaredifferencenote)/2",(differencenote + squaredifferencenote)/2)

        # print(-(averagesquaresum*averagesquaresum)/4 + averagesquaresum)
        # note *= -(averagesquaresum*averagesquaresum)/4 + averagesquaresum
        # note /= 2
        # note = max(0,note)
        # print("anote",note)
        return note

    def evaluateByLines(self):
        lines = self.rewriter.stringlist
        linecount = len(lines)
        totalscore = 0
        for lineindex in range(linecount-1):
            comparisonscore = self.compare_lines(lines[lineindex],lines[lineindex+1])

            totalscore += comparisonscore
            #print("Comparing",lines[lineindex],"with",lines[lineindex+1])
            #print("comparisonscore",comparisonscore)
        distributionscore = 0
        volumepenality = 0
        for line in lines:
            distributionscore += self.analyze_pattern_distribution(line,'Z')['score']
            distributionscore += self.analyze_pattern_distribution(line, 'C')['score']
            volumepenality += self.volumepenality(line)
        return 1-totalscore/linecount, 1-(distributionscore/(linecount*2)),(volumepenality/linecount)


    def compare_lines(self,line1: str, line2: str) -> float:
        """
        Compare two lines and return a similarity score between 0 and 1.
        Uses multiple comparison methods for a comprehensive similarity measure.

        Args:
            line1: First line to compare
            line2: Second line to compare

        Returns:
            float: Similarity score (0 = completely different, 1 = identical)
        """
        # Use SequenceMatcher for sequence similarity
        sequence_similarity = SequenceMatcher(None, line1, line2).ratio()

        # Compare lengths
        length_ratio = min(len(line1), len(line2)) / max(len(line1), len(line2))

        # Compare character frequencies
        def get_char_freq(line):
            freq = {}
            for char in line:
                freq[char] = freq.get(char, 0) + 1
            return freq

        freq1 = get_char_freq(line1)
        freq2 = get_char_freq(line2)

        # Calculate frequency similarity
        all_chars = set(freq1.keys()) | set(freq2.keys())
        freq_similarity = 0
        if all_chars:
            matches = sum(min(freq1.get(c, 0), freq2.get(c, 0)) for c in all_chars)
            total = sum(max(freq1.get(c, 0), freq2.get(c, 0)) for c in all_chars)
            freq_similarity = matches / total

        # Combine similarities with weights
        final_similarity = (0.4 * sequence_similarity +
                            0.3 * length_ratio +
                            0.3 * freq_similarity)

        return final_similarity

    def pattern_score(self,count: int, target: int = 10, spread: float = 5) -> float:
        """
        Calculate a score for pattern occurrence that peaks at the target value
        and decreases on both sides following a Gaussian distribution.

        Args:
            count: Number of pattern occurrences
            target: Optimal number of occurrences (default=10)
            spread: Controls how quickly the score drops off (default=5)

        Returns:
            float: Score between 0 and 1, with 1 being the optimal count
        """
        return np.exp(-((count - target) ** 2) / (2 * spread ** 2))

    def analyze_pattern_distribution(self,text: str, pattern: str) -> dict:
        """
        Analyze the distribution of a pattern in text and return various metrics.

        Args:
            text: Text to analyze
            pattern: Pattern to look for

        Returns:
            dict: Dictionary containing pattern analysis metrics
        """
        count = text.count(pattern)
        score = self.pattern_score(count)

        # Find pattern positions
        positions = []
        start = 0
        while True:
            pos = text.find(pattern, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1

        # Calculate spacing metrics if there are multiple occurrences
        spacing_stats = {}
        if len(positions) > 1:
            spacings = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            spacing_stats = {
                'min_spacing': min(spacings) if spacings else 0,
                'max_spacing': max(spacings) if spacings else 0,
                'avg_spacing': sum(spacings) / len(spacings) if spacings else 0,
                'spacing_variance': np.var(spacings) if spacings else 0
            }

        return {
            'count': count,
            'score': score,
            'positions': positions,
            'density': count / len(text) if text else 0,
            **spacing_stats
        }
    def diversityHOG(self,target_tensor, resize_shape=(128, 128),comparativecount = 40):
        # Preprocess images
        def preprocess_image(img):
            if not len(img.shape)==2:
                # Convert to grayscale
                img = color.rgb2gray(color.rgba2rgb(img))
            # Resize to a consistent size
            return transform.resize(img, resize_shape)

        # Extract HOG features
        def extract_hog_features(img):
            return hog(img, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm='L2-Hys')

        # Preprocess target image
        processed_target = transform.resize(target_tensor.cpu().detach().numpy(), resize_shape)
        #print(processed_target)
        #print(processed_target.shape)

        target_features = extract_hog_features(processed_target)
        #print(target_features)


        if self.HOG_set_features is None:
            path = self.conf.filepath + "*.png"
            image_set_files_names = [i for i in glob.glob(path) if 'bin' in i]
            #print("image_set_files_names1",image_set_files_names)
            if comparativecount < len(image_set_files_names):
                image_set_files_names = random.choices(image_set_files_names,k=comparativecount)

            #print("image_set_files_names2",image_set_files_names)

            image_set = [io.imread(i,as_gray=True) for i in image_set_files_names]
            # if comparativecount > len(image_set):
            #     image_set = random.choices(image_set,k=comparativecount)
            # Preprocess and extract features from image set
            processed_set = [preprocess_image(img) for img in image_set]
            self.HOG_set_features = [extract_hog_features(img) for img in processed_set]

        # Scale features
        scaler = StandardScaler()
        scaled_target = scaler.fit_transform(target_features.reshape(1, -1))
        scaled_set = scaler.transform(np.array(self.HOG_set_features))

        # Calculate Euclidean distances
        distances = euclidean_distances(scaled_target, scaled_set)
        #print(distances)
        return min(distances[0])
        #return statistics.fmean(distances[0])  # Returns distances to each image in the set