#![no_std]

extern crate alloc;

use core::iter::{once, FromIterator};

use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

use log::{info};


fn to_lr_pairs(s: &'_ str) -> impl Iterator<Item = (&'_ str, &'_ str)> {
    s.split_whitespace().flat_map(|w| {
        w.char_indices()
            .skip(1)
            .map(move |(i, _)| (&w[..i], &w[i..]))
            .chain(once((&w[..], &w[0..0])))
    })
}

#[derive(Default)]
struct FeatureCount<'a> {
    pos: usize,
    neg: usize,
    end: usize,
    common: usize,
    all: usize,
    root_features: HashSet<&'a str>,
}

#[derive(PartialEq, Debug)]
pub struct Score<'a> {
    word: &'a str,
    support: usize,
    score: f32,
}

#[derive(PartialEq, Debug)]
pub struct Model<'a> {
    root_set: HashSet<&'a str>,
    pos_set: HashSet<&'a str>,
    neg_set: HashSet<&'a str>,
    common_set: HashSet<&'a str>,

    min_num_of_features: usize,
    min_noun_score: f32,
    max_frequency_when_noun_is_eojeol: usize,
    max_left_bytes: usize,
    max_right_bytes: usize,
    min_noun_frequency: usize,
}
impl<'a> Model<'a> {
    fn new(pos_text: &'a str, neg_text: &'a str) -> Self {
        let pos_set = HashSet::<&str>::from_iter(pos_text.lines().map(|s| s.trim()));
        let neg_set = HashSet::<&str>::from_iter(neg_text.lines().map(|s| s.trim()));
        let common_set: HashSet<&str> = pos_set.intersection(&neg_set).cloned().collect();
        let pos_set: HashSet<&str> = pos_set.difference(&common_set).cloned().collect();
        let neg_set: HashSet<&str> = neg_set.difference(&common_set).cloned().collect();
        let root_set: HashSet<&str> = pos_set
            .iter()
            .cloned()
            .filter(|w| !w.char_indices().any(|(i, _)| pos_set.contains(&w[..i])))
            .chain(
                neg_set
                    .iter()
                    .cloned()
                    .filter(|w| !w.char_indices().any(|(i, _)| neg_set.contains(&w[..i]))),
            )
            .collect();
        Self {
            root_set,
            pos_set,
            neg_set,
            common_set,

            min_num_of_features: 1,
            min_noun_score: 0.3,
            max_frequency_when_noun_is_eojeol: 30,
            max_left_bytes: 20,
            max_right_bytes: 18,
            min_noun_frequency: 1,
        }
    }
    pub fn extract_nouns<'b>(
        &self,
        sents: impl Iterator<Item = &'b str> + Clone,
    ) -> Vec<Score<'b>> {
        let lr_pairs = sents.flat_map(|s| {
            to_lr_pairs(s)
                .filter(|(l, r)| l.len() <= self.max_left_bytes && r.len() <= self.max_right_bytes)
        });
        let mut feature_counts: HashMap<&str, FeatureCount> = HashMap::new();
        info!("count features from lr pairs");
        for (l, r) in lr_pairs {
            let count = feature_counts.entry(l).or_insert(FeatureCount::default());
            count.pos += self.pos_set.contains(r) as usize;
            count.neg += self.neg_set.contains(r) as usize;
            count.common += self.common_set.contains(r) as usize;
            count.all += 1;
            count.end += r.is_empty() as usize;
            if self.root_set.contains(r) {
                count.root_features.insert(r);
            }
        }
        let len = feature_counts.len();
        info!("calculate score of l-terms");
        let prediction_scores = feature_counts.into_iter().enumerate().map(|(i, (word, c))| {
            let base_score = if 0 < c.pos + c.neg {
                (c.pos as f32 - c.neg as f32) / (c.pos + c.neg) as f32
            } else {
                0.0
            };
            let base_support = if base_score > self.min_noun_score {
                c.pos + c.end + c.common
            } else {
                c.neg + c.end + c.common
            };
            let (support, score) = if c.root_features.len() > self.min_num_of_features {
                (base_support, base_score)
            } else if c.all == 0 {
                (base_support, 0.0)
            } else if c.end > self.max_frequency_when_noun_is_eojeol && (c.pos >= c.neg) {
                (base_support, base_score)
            } else if (c.common > 0 || c.pos > 0)
                && (c.end as f32 / c.all as f32 >= 0.3)
                && (c.common >= c.neg)
                || 2 <= c
                    .root_features
                    .iter()
                    .filter(|f| self.pos_set.contains(*f))
                    .count()
            {
                (
                    c.pos + c.common + c.end,
                    (c.pos + c.common + c.end) as f32 / c.all as f32,
                )
            } else {
                (base_support, 0.0)
            };
            if i % 1000 == 0 {
                info!("{}/{}...", i, len);
            }
            Score {word, support, score}
        });
        let mut res: Vec<_> = prediction_scores
            .filter(|s| s.support >= self.min_noun_frequency && s.score >= self.min_noun_score)
            .collect();
        res.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        res
    }
}
impl Default for Model<'_> {
    fn default() -> Self {
        Self::new(
            include_str!("../assets/noun_predictor_ver2_pos.txt"),
            include_str!("../assets/noun_predictor_ver2_neg.txt"),
        )
    }
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use crate::*;
    use std::vec;

    use inline_python::python;

    #[test]
    fn it_to_lr_paris() {
        let s = "가나다";
        let expected = vec![("가", "나다"), ("가나", "다"), ("가나다", "")];
        assert_eq!(to_lr_pairs(&s).collect::<Vec<_>>(), expected);
    }
    #[test]
    fn it_creates_default_feature() {
        Model::default();
    }
    #[test]
    fn it_creates_feature() {
        let pos_text = "에\n에서\n을\n가";
        let neg_text = "그\n그아\n노\n가";
        let feature = Model::new(pos_text, neg_text);
        let expected = Model {
            root_set: HashSet::from_iter(vec!["노", "을", "그", "에"].into_iter()),
            pos_set: HashSet::from_iter(vec!["을", "에", "에서"].into_iter()),
            neg_set: HashSet::from_iter(vec!["노", "그", "그아"].into_iter()),
            common_set: HashSet::from_iter(vec!["가"].into_iter()),
            min_num_of_features: 1,
            min_noun_score: 0.3,
            max_frequency_when_noun_is_eojeol: 30,
            max_left_bytes: 20,
            max_right_bytes: 18,
            min_noun_frequency: 1,
        };
        assert_eq!(feature, expected);
    }
    #[test]
    fn it_extrcats_nouns() {
        let f = Model::default();
        //let test_data = include_str!("../assets/2016-10-20.txt");
        let test_data = "여기는 저기는 거기에 여기다 그렇지 그렇다 그러함 골골골 골골골";
        let res = f.extract_nouns(std::iter::once(test_data));
        let expected = python! {
            from soynlp.utils import DoublespaceLineCorpus
            from soynlp.noun import LRNounExtractor_v2
            noun_extractor = LRNounExtractor_v2(verbose=False)
        }
        assert_eq!( Vec::<Score>::new());
    }
}
