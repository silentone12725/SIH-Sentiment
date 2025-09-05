import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os

def generate_wordcloud_file(file_path, output_file=None, show_plot=True):
    """Generate word cloud from CSV file
    
    Args:
        file_path: Path to CSV file
        output_file: Path to save image (optional)
        show_plot: Whether to display the plot (default: True)
    """
    # Load your dataset
    df = pd.read_csv(file_path, encoding="utf-8")

    # Combine all comments
    text_data = " ".join(df["comment"].astype(str))

    # Extend the default stopwords with domain-specific ones
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update([
        "section", "draft", "provision", "amendment", "company",
        "act", "rule", "subsection", "clause", "law", "mca",
        "shall", "herein", "thereof", "wherein"
    ])

    # Generate word cloud excluding stopwords
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        stopwords=custom_stopwords
    ).generate(text_data)

    # Save the word cloud image locally
    if output_file is None:
        output_file = "mca_wordcloud_filtered.png"
    
    wordcloud.to_file(output_file)
    print(f"✅ Word cloud saved as {output_file} (filtered common terms)")

    # Only show plot if requested
    if show_plot:
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud (Unique Stakeholder Terms)", fontsize=16)
        plt.show()
    
    return wordcloud

if __name__ == "__main__":
    # Only show the plot when running this file directly
    file_path = "mca_consultation_test_data_max_realism_artifacts.csv"
    generate_wordcloud_file(file_path, show_plot=True)
