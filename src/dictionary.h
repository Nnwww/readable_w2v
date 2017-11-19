/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_DICTIONARY_H
#define FASTTEXT_DICTIONARY_H

#include <vector>
#include <string>
#include <istream>
#include <ostream>
#include <random>
#include <memory>
#include <unordered_map>

#include "args.h"
#include "real.h"

namespace fasttext {

struct entry {
  std::string word;
  int64_t count;
};

class Dictionary {
  protected:
    static const int32_t MAX_VOCAB_SIZE = 30000000;
    static const int32_t MAX_LINE_SIZE = 1024;

    int32_t find(const std::string&) const;
    int32_t find(const std::string&, uint32_t h) const;
    void initTableDiscard();
    void reset(std::istream&) const;
    void pushHash(std::vector<int32_t>&, int32_t) const;
    void addSubwords(std::vector<int32_t>&, const std::string&, int32_t) const;

    std::shared_ptr<Args> args_;
    std::vector<int32_t> word2int_;
    std::vector<entry> words_;

    std::vector<real> pdiscard_;
    int32_t nwords_;
    int64_t ntokens_;

    int64_t pruneidx_size_;
    std::unordered_map<int32_t, int32_t> pruneidx_;

   public:
    static const std::string EOS;
    explicit Dictionary(std::shared_ptr<Args>);
    int32_t nwords() const;
    int64_t ntokens() const;
    int32_t getId(const std::string&) const;
    int32_t getId(const std::string&, uint32_t h) const;
    bool discard(int32_t, real) const;
    std::string getWord(int32_t) const;
    uint32_t hash(const std::string& str) const;
    void add(const std::string&);
    bool readWord(std::istream&, std::string&) const;
    void readFromFile(std::istream&);
    std::string getLabel(int32_t) const;
    void save(std::ostream&) const;
    void load(std::istream&);
    int32_t getLine(std::istream&, std::vector<int32_t>&, std::minstd_rand&) const;
    void threshold(int64_t);
    void prune(std::vector<int32_t>&);
    bool isPruned() { return pruneidx_size_ >= 0; }
    std::vector<int64_t> getCounts() const;
};

}

#endif
