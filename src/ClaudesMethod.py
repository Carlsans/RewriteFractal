import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import entropy
import cv2


class ImageComplexityMetrics:
    def __init__(self, device='cuda'):
        self.device = device

    def shannon_entropy(self, tensor):
        """Calculate Shannon entropy of pixel intensities"""
        # Flatten and convert to numpy for histogram
        flat = tensor.flatten().cpu().numpy()
        bins = 256
        # Create histogram with 256 bins for intensity values
        hist, _ = np.histogram(flat, bins=bins, range=(0, 1), density=True)

        # Remove zero probabilities to avoid log(0)
        hist = hist[hist > 0]

        # Calculate Shannon entropy
        return -((np.sum(hist * np.log2(hist)))/np.log2(bins))/bins

    def spatial_entropy(self, tensor, window_size=8):
        """Calculate entropy in local spatial windows"""
        H, W = tensor.shape[-2:]
        entropies = []

        for i in range(0, H - window_size + 1, window_size // 2):
            for j in range(0, W - window_size + 1, window_size // 2):
                window = tensor[i:i + window_size, j:j + window_size]
                if window.numel() > 0:
                    entropies.append(self.shannon_entropy(window))

        return np.mean(entropies) if entropies else 0.0

    def fractal_dimension(self, tensor, max_box_size=None):
        """Estimate fractal dimension using box-counting method"""
        # Convert to binary image using threshold
        binary = (tensor > tensor.mean()).float()

        # Convert to numpy for processing
        img = binary.cpu().numpy().astype(np.uint8)

        if max_box_size is None:
            max_box_size = min(img.shape) // 4

        sizes = []
        counts = []

        for size in range(1, max_box_size + 1):
            # Count boxes that contain at least one foreground pixel
            count = 0
            for i in range(0, img.shape[0], size):
                for j in range(0, img.shape[1], size):
                    box = img[i:i + size, j:j + size]
                    if box.sum() > 0:
                        count += 1

            if count > 0:
                sizes.append(size)
                counts.append(count)

        if len(sizes) < 2:
            return 0.0

        # Fit line to log-log plot
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)

        # Linear regression to find slope
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return -coeffs[0]  # Fractal dimension is negative slope

    def variance_std_metrics(self, tensor):
        """Calculate variance and standard deviation metrics"""
        return {
            'pixel_variance': torch.var(tensor).item(),
            'pixel_std': torch.std(tensor).item(),
        }

    def edge_density(self, tensor):
        """Calculate edge density using Sobel operator"""
        # Add batch and channel dimensions for conv2d
        if len(tensor.shape) == 2:
            tensor_4d = tensor.unsqueeze(0).unsqueeze(0).cuda()

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        # Apply Sobel filters
        grad_x = F.conv2d(tensor_4d, sobel_x, padding=1)
        grad_y = F.conv2d(tensor_4d, sobel_y, padding=1)

        # Calculate gradient magnitude
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Edge density as percentage of pixels above threshold
        threshold = gradient_magnitude.mean() + gradient_magnitude.std()
        edge_pixels = (gradient_magnitude > threshold).float()

        return edge_pixels.mean().item()

    def mutual_information(self, tensor, regions=4):
        """Calculate mutual information between spatial regions"""
        H, W = tensor.shape[-2:]

        # Divide image into regions
        region_h, region_w = H // regions, W // regions

        mi_values = []

        for i in range(regions):
            for j in range(regions):
                if i != j or j != 0:  # Don't compare region with itself
                    # Get two regions
                    r1 = tensor[i * region_h:(i + 1) * region_h, :region_w].flatten()
                    r2 = tensor[j * region_h:(
                                                         j + 1) * region_h, region_w:2 * region_w].flatten() if j * region_h < H and 2 * region_w <= W else r1

                    if len(r1) > 0 and len(r2) > 0:
                        # Convert to numpy and calculate 2D histogram
                        r1_np = r1.cpu().numpy()
                        r2_np = r2.cpu().numpy()

                        # Calculate joint and marginal histograms
                        if np.isnan(r1_np).all() or np.isnan(r2_np).all() :
                            return 0
                        joint_hist, _, _ = np.histogram2d(r1_np, r2_np, bins=16)
                        joint_hist = joint_hist + 1e-10  # Avoid zeros
                        joint_prob = joint_hist / joint_hist.sum()

                        marginal1 = joint_prob.sum(axis=1)
                        marginal2 = joint_prob.sum(axis=0)

                        # Calculate mutual information
                        mi = 0
                        for x in range(len(marginal1)):
                            for y in range(len(marginal2)):
                                if joint_prob[x, y] > 0:
                                    mi += joint_prob[x, y] * np.log2(joint_prob[x, y] / (marginal1[x] * marginal2[y]))

                        mi_values.append(mi)

        return np.mean(mi_values)/regions if mi_values else 0.0

    def compression_complexity(self, tensor):
        """Approximate Kolmogorov complexity using compression ratio"""
        # Convert to uint8 for compression
        img_uint8 = (tensor.cpu().numpy() * 255).astype(np.uint8)

        # Encode as PNG (lossless compression)
        _, encoded = cv2.imencode('.png', img_uint8)
        compressed_size = len(encoded)

        original_size = tensor.numel()

        # Compression ratio (lower ratio = higher complexity)
        compression_ratio = compressed_size / original_size

        # Complexity score (inverse of compression efficiency)
        return 1.0 / compression_ratio if compression_ratio > 0 else 0.0

    def glcm_features(self, tensor, distances=[1], angles=[0, 45, 90, 135]):
        """Calculate Gray-Level Co-occurrence Matrix features"""
        # Convert to uint8 with reduced levels for GLCM
        img = (tensor.cpu().numpy() * 63).astype(np.uint8)  # 64 gray levels

        contrast_values = []
        homogeneity_values = []
        energy_values = []

        for distance in distances:
            for angle in angles:
                # Calculate GLCM manually (simplified version)
                glcm = self._calculate_glcm(img, distance, angle)

                # Normalize GLCM
                glcm = glcm / (glcm.sum() + 1e-10)

                # Calculate features
                contrast = self._glcm_contrast(glcm)
                homogeneity = self._glcm_homogeneity(glcm)
                energy = self._glcm_energy(glcm)

                contrast_values.append(contrast)
                homogeneity_values.append(homogeneity)
                energy_values.append(energy)

        return {
            'glcm_contrast': np.mean(contrast_values),
            'glcm_homogeneity': np.mean(homogeneity_values),
            'glcm_energy': np.mean(energy_values)
        }

    def _calculate_glcm(self, img, distance, angle):
        """Simplified GLCM calculation"""
        rows, cols = img.shape
        max_gray = 64  # Reduced gray levels
        glcm = np.zeros((max_gray, max_gray))

        # Direction vectors for different angles
        if angle == 0:
            di, dj = 0, distance
        elif angle == 45:
            di, dj = -distance, distance
        elif angle == 90:
            di, dj = -distance, 0
        else:  # 135
            di, dj = -distance, -distance

        for i in range(rows):
            for j in range(cols):
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    glcm[img[i, j], img[ni, nj]] += 1

        return glcm

    def _glcm_contrast(self, glcm):
        """Calculate contrast from GLCM"""
        contrast = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                contrast += glcm[i, j] * (i - j) ** 2
        return contrast

    def _glcm_homogeneity(self, glcm):
        """Calculate homogeneity from GLCM"""
        homogeneity = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                homogeneity += glcm[i, j] / (1 + abs(i - j))
        return homogeneity

    def _glcm_energy(self, glcm):
        """Calculate energy from GLCM"""
        return np.sum(glcm ** 2)

    def local_binary_patterns_variance(self, tensor, radius=1, n_points=8):
        """Calculate Local Binary Pattern variance"""
        img = tensor.cpu().numpy()

        # Simple LBP implementation
        lbp_variance = 0
        count = 0

        for i in range(radius, img.shape[0] - radius):
            for j in range(radius, img.shape[1] - radius):
                center = img[i, j]
                pattern = []

                # Sample points in circle
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))

                    if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                        pattern.append(1 if img[x, y] >= center else 0)

                if pattern:
                    # Calculate variance of pattern
                    lbp_variance += np.var(pattern)
                    count += 1

        return lbp_variance / count if count > 0 else 0.0

    def wavelet_coefficient_stats(self, tensor):
        """Calculate wavelet coefficient statistics (simplified)"""
        # Simple 2D Haar wavelet transform using conv2d
        if len(tensor.shape) == 2:
            tensor_4d = tensor.unsqueeze(0).unsqueeze(0)

        # Haar wavelet kernels
        haar_ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=self.device).unsqueeze(0).unsqueeze(0)
        haar_lh = torch.tensor([[0.5, -0.5], [0.5, -0.5]], device=self.device).unsqueeze(0).unsqueeze(0)
        haar_hl = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], device=self.device).unsqueeze(0).unsqueeze(0)
        haar_hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device=self.device).unsqueeze(0).unsqueeze(0)

        # Apply filters with stride 2 for downsampling
        ll = F.conv2d(tensor_4d, haar_ll, stride=2)
        lh = F.conv2d(tensor_4d, haar_lh, stride=2)
        hl = F.conv2d(tensor_4d, haar_hl, stride=2)
        hh = F.conv2d(tensor_4d, haar_hh, stride=2)

        return {
            'wavelet_ll_variance': torch.var(ll).item(),
            'wavelet_lh_variance': torch.var(lh).item(),
            'wavelet_hl_variance': torch.var(hl).item(),
            'wavelet_hh_variance': torch.var(hh).item(),
            'wavelet_entropy_lh': self.shannon_entropy(lh.squeeze()),
            'wavelet_entropy_hl': self.shannon_entropy(hl.squeeze()),
            'wavelet_entropy_hh': self.shannon_entropy(hh.squeeze())
        }

    def color_diversity_metrics(self, tensor, n_bins=64):
        """Calculate color diversity metrics (for grayscale, intensity diversity)"""
        # Histogram entropy
        hist = torch.histc(tensor, bins=n_bins, min=0, max=1)
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros

        color_entropy = -torch.sum(hist * torch.log2(hist + 1e-10)).item()

        # Unique intensity levels
        unique_values = torch.unique(tensor)
        complexity_index = len(unique_values) / tensor.numel()

        return {
            'color_histogram_entropy': color_entropy,
            'intensity_complexity_index': complexity_index,
            'unique_intensity_ratio': len(unique_values) / n_bins
        }

    def calculate_all_metrics(self, tensor):
        """Calculate all complexity metrics for the input tensor"""
        tensor = tensor.to(self.device)

        results = {}

        # Basic entropy measures
        results['shannon_entropy'] = self.shannon_entropy(tensor)
        results['spatial_entropy'] = self.spatial_entropy(tensor)

        # Complexity measures
        results['fractal_dimension'] = self.fractal_dimension(tensor)
        results.update(self.variance_std_metrics(tensor))
        results['edge_density'] = self.edge_density(tensor)

        # Information-theoretic measures
        results['mutual_information'] = self.mutual_information(tensor)
        results['compression_complexity'] = self.compression_complexity(tensor)

        # Texture analysis
        results.update(self.glcm_features(tensor))
        results['lbp_variance'] = self.local_binary_patterns_variance(tensor)

        # Frequency domain
        results.update(self.wavelet_coefficient_stats(tensor))

        # Color/intensity diversity
        results.update(self.color_diversity_metrics(tensor))

        return results


# Example usage
def demo_complexity_metrics():
    # Create sample 2D tensor (grayscale image)
    torch.manual_seed(42)

    # Create different types of test images
    images = {
        'random_noise': torch.rand(128, 128),
        'gradient': torch.linspace(0, 1, 128).unsqueeze(0).repeat(128, 1),
        'checkerboard': torch.zeros(128, 128),
        'fractal_like': torch.rand(128, 128)
    }

    # Create checkerboard pattern
    for i in range(0, 128, 16):
        for j in range(0, 128, 16):
            if (i // 16 + j // 16) % 2 == 0:
                images['checkerboard'][i:i + 16, j:j + 16] = 1.0

    # Create fractal-like pattern
    x = torch.linspace(-2, 2, 128)
    y = torch.linspace(-2, 2, 128)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    images['fractal_like'] = torch.sin(X * Y) * torch.cos(X ** 2 + Y ** 2)
    images['fractal_like'] = (images['fractal_like'] + 1) / 2  # Normalize to [0, 1]

    # Initialize metrics calculator
    metrics_calc = ImageComplexityMetrics()

    # Calculate metrics for each image
    for name, img in images.items():
        print(f"\n=== Metrics for {name.upper()} ===")
        metrics = metrics_calc.calculate_all_metrics(img)

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    demo_complexity_metrics()