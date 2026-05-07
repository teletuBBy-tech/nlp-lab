from sample_transcript import sample_transcript

def get_transcript(url):
    return " ".join([t["text"] for t in sample_transcript])

def get_transcript_with_time(url):
    return sample_transcript