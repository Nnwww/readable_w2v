/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "dictionary.h"

#include <assert.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <cmath>

namespace fasttext {
  const std::string Dictionary::EOS = "</s>";

  Dictionary::Dictionary(std::shared_ptr<Args> args) :
    args_(args), word2int_(MAX_VOCAB_SIZE, -1), nwords_(0), nlabels_(0), ntokens_(0), pruneidx_size_(-1) {}

  int32_t Dictionary::find(const std::string& w) const {
    return find(w, hash(w));
  }

  int32_t Dictionary::find(const std::string& w, uint32_t h) const {
    int32_t id = h % MAX_VOCAB_SIZE;
    while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
      id = (id + 1) % MAX_VOCAB_SIZE;
    }
    return id;
  }

  void Dictionary::add(const std::string& w) {
    int32_t h = find(w);
    ntokens_++;
    if (word2int_[h] == -1) {
      entry e;
      e.word = w;
      e.count = 1;
      e.type = getType(w);
      words_.push_back(e);
      word2int_[h] = size_++;
    } else {
      words_[word2int_[h]].count++;
    }
  }

  int32_t Dictionary::nwords() const {
    return nwords_;
  }

  int32_t Dictionary::nlabels() const {
    return nlabels_;
  }

  int64_t Dictionary::ntokens() const {
    return ntokens_;
  }

  bool Dictionary::discard(int32_t id, real rand) const {
    assert(id >= 0);
    assert(id < nwords_);
    if (args_->model == model_name::sup) return false;
    return rand > pdiscard_[id];
  }

  int32_t Dictionary::getId(const std::string& w, uint32_t h) const {
    int32_t id = find(w, h);
    return word2int_[id];
  }

  int32_t Dictionary::getId(const std::string& w) const {
    int32_t h = find(w);
    return word2int_[h];
  }

  entry_type Dictionary::getType(id_t id) const {
    assert(id >= 0);
    assert(id < size_);
    return words_[id].type;
  }

  entry_type Dictionary::getType(const std::string& w) const {
    return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
  }


  std::string Dictionary::getWord(int32_t id) const {
    assert(id >= 0);
    assert(id < size_);
    return words_[id].word;
  }

  uint32_t Dictionary::hash(const std::string& str) const {
    uint32_t h = 2166136261;
    for (size_t i = 0; i < str.size(); i++) {
      h = h ^ uint32_t(str[i]);
      h = h * 16777619;
    }
    return h;
  }

  bool Dictionary::readWord(std::istream& in, std::string& word) const
  {
    char c;
    std::streambuf& sb = *in.rdbuf();
    word.clear();
    while ((c = sb.sbumpc()) != EOF) {
      if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
          c == '\f' || c == '\0') {
        if (word.empty()) {
          if (c == '\n') {
            word += EOS;
            return true;
          }
          continue;
        } else {
          if (c == '\n')
            sb.sungetc();
          return true;
        }
      }
      word.push_back(c);
    }
    // trigger eofbit
    in.get();
    return !word.empty();
  }

  void Dictionary::readFromFile(std::istream& in) {
    std::string word;
    int64_t minThreshold = 1;
    while (readWord(in, word)) {
      add(word);
      if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
        std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::flush;
      }
      if (size_ > 0.75 * MAX_VOCAB_SIZE) {
        minThreshold++;
        threshold(minThreshold, minThreshold);
      }
    }
    threshold(args_->minCount, args_->minCountLabel);
    initTableDiscard();
    if (args_->verbose > 0) {
      std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::endl;
      std::cerr << "Number of words:  " << nwords_ << std::endl;
      std::cerr << "Number of labels: " << nlabels_ << std::endl;
    }
    if (size_ == 0) {
      std::cerr << "Empty vocabulary. Try a smaller -minCount value."
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  void Dictionary::threshold(int64_t t, int64_t tl) {
    sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
        if (e1.type != e2.type) return e1.type < e2.type;
        return e1.count > e2.count;
      });
    words_.erase(remove_if(words_.begin(), words_.end(), [&](const entry& e) {
          return (e.type == entry_type::word && e.count < t) ||
            (e.type == entry_type::label && e.count < tl);
        }), words_.end());
    words_.shrink_to_fit();
    size_ = 0;
    nwords_ = 0;
    nlabels_ = 0;
    std::fill(word2int_.begin(), word2int_.end(), -1);
    for (auto it = words_.begin(); it != words_.end(); ++it) {
      int32_t h = find(it->word);
      word2int_[h] = size_++;
      if (it->type == entry_type::word) nwords_++;
      if (it->type == entry_type::label) nlabels_++;
    }
  }


  void Dictionary::initTableDiscard() {
    pdiscard_.resize(size_);
    for (int32_t i = 0; i < size_; i++) {
      real f = real(words_[i].count) / real(ntokens_);
      pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
    }
  }


  void Dictionary::reset(std::istream& in) const {
    if (in.eof()) {
      in.clear();
      in.seekg(std::streampos(0));
    }
  }

  int32_t Dictionary::getLine(std::istream& in,
                              std::vector<int32_t>& words,
                              std::minstd_rand& rng) const {
    std::uniform_real_distribution<> uniform(0, 1);
    std::string token;
    int32_t ntokens = 0;

    reset(in);
    words.clear();
    while (readWord(in, token)) {
      int32_t h = find(token);
      int32_t wid = word2int_[h];
      if (wid < 0) continue;

      ntokens++;
      if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
        words.push_back(wid);
      }
      if (ntokens > MAX_LINE_SIZE || token == EOS) break;
    }
    return ntokens;
  }

  int32_t Dictionary::getLine(std::istream& in,
                              std::vector<int32_t>& words,
                              std::vector<int32_t>& labels,
                              std::minstd_rand& rng) const {
    std::string token;
    int32_t ntokens = 0;

    reset(in);
    words.clear();
    labels.clear();
    while (readWord(in, token)) {
      uint32_t h = hash(token);
      int32_t wid = getId(token, h);
      entry_type type = wid < 0 ? getType(token) : getType(wid);

      ntokens++;
      if (type == entry_type::word) {
        words.push_back(wid);
      } else if (type == entry_type::label && wid >= 0) {
        labels.push_back(wid - nwords_);
      }
      if (token == EOS) break;
    }
    return ntokens;
  }

  void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
    if (pruneidx_size_ == 0 || id < 0) return;
    if (pruneidx_size_ > 0) {
      if (pruneidx_.count(id)) {
        id = pruneidx_.at(id);
      } else {
        return;
      }
    }
    hashes.push_back(nwords_ + id);
  }

  std::string Dictionary::getLabel(int32_t lid) const {
    if (lid < 0 || lid >= nlabels_) {
      throw std::invalid_argument("Label id is out of range [0, " + std::to_string(nlabels_) + "]");
    }
    return words_[lid + nwords_].word;
  }

  void Dictionary::save(std::ostream& out) const {
    out.write((char*) &size_, sizeof(int32_t));
    out.write((char*) &nwords_, sizeof(int32_t));
    out.write((char*) &nlabels_, sizeof(int32_t));
    out.write((char*) &ntokens_, sizeof(int64_t));
    out.write((char*) &pruneidx_size_, sizeof(int64_t));
    for (int32_t i = 0; i < size_; i++) {
      entry e = words_[i];
      out.write(e.word.data(), e.word.size() * sizeof(char));
      out.put(0);
      out.write((char*) &(e.count), sizeof(int64_t));
      out.write((char*) &(e.type), sizeof(entry_type));
    }
    for (const auto pair : pruneidx_) {
      out.write((char*) &(pair.first), sizeof(int32_t));
      out.write((char*) &(pair.second), sizeof(int32_t));
    }
  }

  void Dictionary::load(std::istream& in) {
    words_.clear();
    std::fill(word2int_.begin(), word2int_.end(), -1);
    in.read((char*) &size_, sizeof(int32_t));
    in.read((char*) &nwords_, sizeof(int32_t));
    in.read((char*) &nlabels_, sizeof(int32_t));
    in.read((char*) &ntokens_, sizeof(int64_t));
    in.read((char*) &pruneidx_size_, sizeof(int64_t));
    for (int32_t i = 0; i < size_; i++) {
      char c;
      entry e;
      while ((c = in.get()) != 0) {
        e.word.push_back(c);
      }
      in.read((char*) &(e.count), sizeof(int64_t));
      in.read((char*) &(e.type), sizeof(entry_type));
      words_.push_back(e);
      word2int_[find(e.word)] = i;
    }
    pruneidx_.clear();
    for (int32_t i = 0; i < pruneidx_size_; i++) {
      int32_t first;
      int32_t second;
      in.read((char*) &first, sizeof(int32_t));
      in.read((char*) &second, sizeof(int32_t));
      pruneidx_[first] = second;
    }
    initTableDiscard();
  }

  void Dictionary::prune(std::vector<int32_t>& idx) {
    std::vector<int32_t> words;
    for (auto it = idx.cbegin(); it != idx.cend(); ++it) {
      if (*it < nwords_) {words.push_back(*it);}
    }
    std::sort(words.begin(), words.end());
    idx = words;

    pruneidx_size_ = pruneidx_.size();

    std::fill(word2int_.begin(), word2int_.end(), -1);
    size_t j = 0;
    for (size_t i = 0; i < words_.size(); i++) {
      if (getType(i) == entry_type::label ||
          (j < words.size() && words[j] == static_cast<int32_t>(i))) {
        words_[j] = words_[i];
        word2int_[find(words_[j].word)] = j;
        j++;
      }
    }
    nwords_ = words.size();
    size_ = nwords_ +  nlabels_;
  }

  std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
    std::vector<int64_t> counts;
    for (const auto& w : words_) {
      if (w.type == type) counts.push_back(w.count);
    }
    return counts;
  }
}
