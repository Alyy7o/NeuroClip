import csv
import os

# Create dataset directory
os.makedirs("kaggle_eval_dataset", exist_ok=True)

CSV_FILENAME = "kaggle_eval_dataset/summarization_eval_pack.csv"

# Designing the 50 query-video cases across 10 academic domains as mentioned in the research paper.
# This dataset is designed to be fed into the automated testing script to evaluate the Kaggle backend.
#
# VIDEO SELECTION CRITERIA (updated 2026-05-01):
#   - All videos MUST be under 30 minutes to avoid ngrok gateway timeouts
#   - All videos MUST be publicly available (not region-locked or removed)
#   - Prefer channels with stable content (3Blue1Brown, freeCodeCamp shorts, Khan Academy, etc.)
EVAL_DATA = [
    # ---- Domain: Computer Science (16 queries) ----
    # Video: "Object-oriented Programming in 7 minutes" by Mosh (7 min) — replaces 8-hour Data Structures course
    {"video_url": "https://www.youtube.com/watch?v=pTB0EiLXUC8", "domain": "Computer Science", "query": "what is object oriented programming", "difficulty": "Easy", "gt_start": 10, "gt_end": 60},
    {"video_url": "https://www.youtube.com/watch?v=pTB0EiLXUC8", "domain": "Computer Science", "query": "explain inheritance and polymorphism", "difficulty": "Medium", "gt_start": 180, "gt_end": 300},
    {"video_url": "https://www.youtube.com/watch?v=pTB0EiLXUC8", "domain": "Computer Science", "query": "what are the four pillars of OOP", "difficulty": "Easy", "gt_start": 60, "gt_end": 120},
    {"video_url": "https://www.youtube.com/watch?v=pTB0EiLXUC8", "domain": "Computer Science", "query": "how encapsulation protects data state", "difficulty": "Hard", "gt_start": 120, "gt_end": 200},

    # Video: "Hash Tables and Hash Functions" by Computer Science (10 min) — same as before, confirmed available
    {"video_url": "https://www.youtube.com/watch?v=v4cd1O4zkGw", "domain": "Computer Science", "query": "what is a hash table", "difficulty": "Easy", "gt_start": 10, "gt_end": 60},
    {"video_url": "https://www.youtube.com/watch?v=v4cd1O4zkGw", "domain": "Computer Science", "query": "how to handle hash collisions", "difficulty": "Medium", "gt_start": 150, "gt_end": 220},
    {"video_url": "https://www.youtube.com/watch?v=v4cd1O4zkGw", "domain": "Computer Science", "query": "time complexity of hash map lookups", "difficulty": "Medium", "gt_start": 80, "gt_end": 130},
    {"video_url": "https://www.youtube.com/watch?v=v4cd1O4zkGw", "domain": "Computer Science", "query": "open addressing versus chaining", "difficulty": "Hard", "gt_start": 250, "gt_end": 350},

    # Video: "APIs for Beginners" by freeCodeCamp (shortened alt) — replaces unavailable bqwJzMqDWXg
    {"video_url": "https://www.youtube.com/watch?v=GZvSYJDk-us", "domain": "Computer Science", "query": "what is an API", "difficulty": "Easy", "gt_start": 20, "gt_end": 80},
    {"video_url": "https://www.youtube.com/watch?v=GZvSYJDk-us", "domain": "Computer Science", "query": "difference between REST and SOAP", "difficulty": "Medium", "gt_start": 180, "gt_end": 260},
    {"video_url": "https://www.youtube.com/watch?v=GZvSYJDk-us", "domain": "Computer Science", "query": "HTTP status codes explained", "difficulty": "Medium", "gt_start": 300, "gt_end": 380},
    {"video_url": "https://www.youtube.com/watch?v=GZvSYJDk-us", "domain": "Computer Science", "query": "API authentication methods", "difficulty": "Medium", "gt_start": 400, "gt_end": 480},

    # Video: "Distributed Systems in One Lesson" by Tim Berglund (shortened alt) — replaces unavailable mXTwvZV0ns4
    {"video_url": "https://www.youtube.com/watch?v=Y6Ev8GIlbxc", "domain": "Computer Science", "query": "what is a distributed system", "difficulty": "Easy", "gt_start": 15, "gt_end": 90},
    {"video_url": "https://www.youtube.com/watch?v=Y6Ev8GIlbxc", "domain": "Computer Science", "query": "what is the CAP theorem", "difficulty": "Easy", "gt_start": 120, "gt_end": 200},
    {"video_url": "https://www.youtube.com/watch?v=Y6Ev8GIlbxc", "domain": "Computer Science", "query": "horizontal vs vertical scaling", "difficulty": "Easy", "gt_start": 250, "gt_end": 320},
    {"video_url": "https://www.youtube.com/watch?v=Y6Ev8GIlbxc", "domain": "Computer Science", "query": "how eventual consistency works", "difficulty": "Easy", "gt_start": 350, "gt_end": 420},

    # ---- Domain: Artificial Intelligence (8 queries) ----
    # Video: 3Blue1Brown "But what is a neural network?" (~19 min) — confirmed available
    {"video_url": "https://www.youtube.com/watch?v=aircAruvnKk", "domain": "Artificial Intelligence", "query": "what is a neural network", "difficulty": "Easy", "gt_start": 45, "gt_end": 120},
    {"video_url": "https://www.youtube.com/watch?v=aircAruvnKk", "domain": "Artificial Intelligence", "query": "how activation functions work", "difficulty": "Medium", "gt_start": 200, "gt_end": 280},
    {"video_url": "https://www.youtube.com/watch?v=aircAruvnKk", "domain": "Artificial Intelligence", "query": "what is the cost function", "difficulty": "Medium", "gt_start": 350, "gt_end": 450},
    {"video_url": "https://www.youtube.com/watch?v=aircAruvnKk", "domain": "Artificial Intelligence", "query": "gradient descent explanation", "difficulty": "Hard", "gt_start": 500, "gt_end": 620},

    # Video: 3Blue1Brown "Gradient descent, how neural networks learn" (~21 min) — replaces unavailable I6B_n1P-Ums
    {"video_url": "https://www.youtube.com/watch?v=IHZwWFHWa-w", "domain": "Artificial Intelligence", "query": "what is backpropagation", "difficulty": "Easy", "gt_start": 30, "gt_end": 110},
    {"video_url": "https://www.youtube.com/watch?v=IHZwWFHWa-w", "domain": "Artificial Intelligence", "query": "chain rule in neural networks", "difficulty": "Hard", "gt_start": 180, "gt_end": 300},
    {"video_url": "https://www.youtube.com/watch?v=IHZwWFHWa-w", "domain": "Artificial Intelligence", "query": "calculating error derivatives", "difficulty": "Medium", "gt_start": 350, "gt_end": 420},
    {"video_url": "https://www.youtube.com/watch?v=IHZwWFHWa-w", "domain": "Artificial Intelligence", "query": "updating weights and biases", "difficulty": "Easy", "gt_start": 480, "gt_end": 550},

    # ---- Domain: Mathematics (8 queries) ----
    {"video_url": "https://www.youtube.com/watch?v=HZGCoVF3YvM", "domain": "Mathematics", "query": "what is Bayes theorem", "difficulty": "Easy", "gt_start": 20, "gt_end": 90},
    {"video_url": "https://www.youtube.com/watch?v=HZGCoVF3YvM", "domain": "Mathematics", "query": "prior and posterior probabilities", "difficulty": "Medium", "gt_start": 150, "gt_end": 220},
    {"video_url": "https://www.youtube.com/watch?v=HZGCoVF3YvM", "domain": "Mathematics", "query": "conditional probability definition", "difficulty": "Easy", "gt_start": 280, "gt_end": 350},
    {"video_url": "https://www.youtube.com/watch?v=HZGCoVF3YvM", "domain": "Mathematics", "query": "false positive paradox explanation", "difficulty": "Hard", "gt_start": 400, "gt_end": 520},
    
    {"video_url": "https://www.youtube.com/watch?v=fNk_zzaMoSs", "domain": "Mathematics", "query": "what is a derivative", "difficulty": "Easy", "gt_start": 30, "gt_end": 100},
    {"video_url": "https://www.youtube.com/watch?v=fNk_zzaMoSs", "domain": "Mathematics", "query": "slope of the tangent line", "difficulty": "Easy", "gt_start": 140, "gt_end": 210},
    {"video_url": "https://www.youtube.com/watch?v=fNk_zzaMoSs", "domain": "Mathematics", "query": "power rule in calculus", "difficulty": "Easy", "gt_start": 250, "gt_end": 320},
    {"video_url": "https://www.youtube.com/watch?v=fNk_zzaMoSs", "domain": "Mathematics", "query": "limit definition of a derivative", "difficulty": "Hard", "gt_start": 380, "gt_end": 480},

    # ---- Domain: Physics (3 queries) ----
    {"video_url": "https://www.youtube.com/watch?v=YmEKGGivJQU", "domain": "Physics", "query": "conservation of momentum", "difficulty": "Easy", "gt_start": 40, "gt_end": 110},
    {"video_url": "https://www.youtube.com/watch?v=YmEKGGivJQU", "domain": "Physics", "query": "elastic vs inelastic collisions", "difficulty": "Easy", "gt_start": 150, "gt_end": 230},
    {"video_url": "https://www.youtube.com/watch?v=YmEKGGivJQU", "domain": "Physics", "query": "calculating total impulse", "difficulty": "Easy", "gt_start": 280, "gt_end": 350},

    # ---- Domain: Cybersecurity (3 queries) ----
    {"video_url": "https://www.youtube.com/watch?v=inWWhr5tnEA", "domain": "Cybersecurity", "query": "what is a buffer overflow", "difficulty": "Easy", "gt_start": 20, "gt_end": 90},
    {"video_url": "https://www.youtube.com/watch?v=inWWhr5tnEA", "domain": "Cybersecurity", "query": "how memory stacks work", "difficulty": "Easy", "gt_start": 120, "gt_end": 200},
    {"video_url": "https://www.youtube.com/watch?v=inWWhr5tnEA", "domain": "Cybersecurity", "query": "preventing stack overflow attacks", "difficulty": "Easy", "gt_start": 300, "gt_end": 380},

    # ---- Domain: Economics (3 queries) ----
    {"video_url": "https://www.youtube.com/watch?v=g9aDizJpd_s", "domain": "Economics", "query": "supply and demand curve", "difficulty": "Easy", "gt_start": 30, "gt_end": 100},
    {"video_url": "https://www.youtube.com/watch?v=g9aDizJpd_s", "domain": "Economics", "query": "what is market equilibrium", "difficulty": "Medium", "gt_start": 150, "gt_end": 220},
    {"video_url": "https://www.youtube.com/watch?v=g9aDizJpd_s", "domain": "Economics", "query": "price elasticity of demand", "difficulty": "Medium", "gt_start": 280, "gt_end": 360},

    # ---- Domain: Biology (3 queries) ----
    {"video_url": "https://www.youtube.com/watch?v=8kK2zwjRV0M", "domain": "Biology", "query": "what is cellular respiration", "difficulty": "Easy", "gt_start": 20, "gt_end": 90},
    {"video_url": "https://www.youtube.com/watch?v=8kK2zwjRV0M", "domain": "Biology", "query": "glycolysis process explained", "difficulty": "Easy", "gt_start": 130, "gt_end": 210},
    {"video_url": "https://www.youtube.com/watch?v=8kK2zwjRV0M", "domain": "Biology", "query": "krebs cycle overview", "difficulty": "Easy", "gt_start": 280, "gt_end": 350},

    # ---- Domain: Chemistry (2 queries) ----
    {"video_url": "https://www.youtube.com/watch?v=a8CGsroSqFs", "domain": "Chemistry", "query": "covalent vs ionic bonds", "difficulty": "Easy", "gt_start": 40, "gt_end": 120},
    {"video_url": "https://www.youtube.com/watch?v=a8CGsroSqFs", "domain": "Chemistry", "query": "how electrons are shared", "difficulty": "Easy", "gt_start": 180, "gt_end": 250},

    # ---- Domain: Engineering (2 queries) ----
    {"video_url": "https://www.youtube.com/watch?v=cM_XjQzIfJ0", "domain": "Engineering", "query": "how a combustion engine works", "difficulty": "Easy", "gt_start": 25, "gt_end": 95},
    {"video_url": "https://www.youtube.com/watch?v=cM_XjQzIfJ0", "domain": "Engineering", "query": "four stroke engine cycle", "difficulty": "Easy", "gt_start": 130, "gt_end": 210},

    # ---- Domain: Statistics (2 queries) ----
    {"video_url": "https://www.youtube.com/watch?v=YAlJCIGH2uQ", "domain": "Statistics", "query": "standard deviation explained", "difficulty": "Medium", "gt_start": 45, "gt_end": 115},
    {"video_url": "https://www.youtube.com/watch?v=YAlJCIGH2uQ", "domain": "Statistics", "query": "calculating variance from mean", "difficulty": "Hard", "gt_start": 180, "gt_end": 260},
]

def generate_csv():
    print(f"Generating large-scale dataset: {CSV_FILENAME}")
    with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "video_url", "domain", "query", "difficulty", "gt_start", "gt_end"])
        for i, data in enumerate(EVAL_DATA):
            # Format: id, video_url, domain, query, difficulty, gt_start, gt_end
            writer.writerow([
                f"Q_{str(i+1).zfill(3)}",
                data["video_url"],
                data["domain"],
                data["query"],
                data["difficulty"],
                data["gt_start"],
                data["gt_end"]
            ])
    
    # Print summary
    unique_videos = list(dict.fromkeys(d["video_url"] for d in EVAL_DATA))
    print(f"Dataset successfully created with {len(EVAL_DATA)} query-video pairs!")
    print(f"  Unique videos: {len(unique_videos)}")
    for v in unique_videos:
        count = sum(1 for d in EVAL_DATA if d["video_url"] == v)
        print(f"    {v} ({count} queries)")

if __name__ == "__main__":
    generate_csv()
