require 'torch'
require 'nn'

require 'VanillaRNN'
require 'LSTM'

local utils = require 'util.utils'


local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')
-- For languages in UTF8
local utf8 = require 'lua-utf8'

function LM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
  end

  self.model_type = utils.get_kwarg(kwargs, 'model_type')
  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_size')
  self.rnn_size = utils.get_kwarg(kwargs, 'rnn_size')
  self.num_layers = utils.get_kwarg(kwargs, 'num_layers')
  self.dropout = utils.get_kwarg(kwargs, 'dropout')
  self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')

  local V, D, H = self.vocab_size, self.wordvec_dim, self.rnn_size

  self.net = nn.Sequential()
  self.rnns = {}
  self.bn_view_in = {}
  self.bn_view_out = {}

  self.net:add(nn.LookupTable(V, D))
  for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    if self.model_type == 'rnn' then
      rnn = nn.VanillaRNN(prev_dim, H)
    elseif self.model_type == 'lstm' then
      rnn = nn.LSTM(prev_dim, H)
    end
    rnn.remember_states = true
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
    if self.batchnorm == 1 then
      local view_in = nn.View(1, 1, -1):setNumInputDims(3)
      table.insert(self.bn_view_in, view_in)
      self.net:add(view_in)
      self.net:add(nn.BatchNormalization(H))
      local view_out = nn.View(1, -1):setNumInputDims(2)
      table.insert(self.bn_view_out, view_out)
      self.net:add(view_out)
    end
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
  end

  -- After all the RNNs run, we will have a tensor of shape (N, T, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(self.view1)
  self.net:add(nn.Linear(H, V))
  self.net:add(self.view2)
end


function LM:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)

  for _, view_in in ipairs(self.bn_view_in) do
    view_in:resetSize(N * T, -1)
  end
  for _, view_out in ipairs(self.bn_view_out) do
    view_out:resetSize(N, T, -1)
  end

  return self.net:forward(input)
end


function LM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end


function LM:parameters()
  return self.net:parameters()
end


function LM:training()
  self.net:training()
  parent.training(self)
end


function LM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end


function LM:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end


function LM:encode_string(s)
  local encoded = torch.LongTensor(utf8.len(s))
  for i = 1, utf8.len(s) do
    local token = utf8.sub(s,i, i)
    local idx = self.token_to_idx[token]
    assert(idx ~= nil, 'Got invalid idx')
    encoded[i] = idx
  end
  return encoded
end


function LM:decode_string(encoded)
  assert(torch.isTensor(encoded) and encoded:dim() == 1)
  local s = ''
  for i = 1, encoded:size(1) do
    local idx = encoded[i]
    local token = self.idx_to_token[idx]
    s = s .. token
  end
  return s
end


--[[
RS edit

Sample from the language model until it reaches a terminator character.

Inputs:
- start_text: string, can be ""
- terminator_chars: class of chars in Lua match format, e.g. "[!?\\.]" (note the double escape, ugh)
- min_num_words: if terminator char reached before this threshold, keep going until the next one

Note temperature table; probably worth fiddling with.

Returns:
- the generated string!
--]]

function LM:sample(start_text, terminator_chars, min_num_words)
  self:resetStates()
  local scores

  if #start_text > 0 then
    -- warm up model with start text (but don't add to sampled string)
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2) -- RS: I definitely do not understand this part
    scores = self:forward(x)[{{}, {T0, T0}}]
  else
    local w = self.net:get(1).weight
    scores = w.new(1, 1, self.vocab_size):fill(1)
  end

  local terminated = false
  local num_words_approx = 1

  local temps = {0.5, 0.6, 0.7, 0.8, 0.9}
  local temp = temps[math.random(#temps)] -- for this run

  local next_char_idx = nil
  local next_char = nil
  local sampled_string = ''

  local max_length_to_generate = 140 -- seems reasonable

  while (not terminated) and (#sampled_string < max_length_to_generate) do

    local probs = torch.div(scores, temp):double():exp():squeeze()
    probs:div(torch.sum(probs))
    next_char_idx = torch.multinomial(probs, 1):view(1, 1)
    scores = self:forward(next_char_idx)

    next_char = self.idx_to_token[next_char_idx[1][1]]

    -- sampled_text:resize(1, length_so_far)
    -- sampled_text[{{}, {length_so_far, length_so_far}}]:copy(next_char)

    sampled_string = sampled_string .. next_char

    if next_char == ' ' then
      num_words_approx = num_words_approx + 1 -- close enough
    end

    if next_char:match(terminator_chars) then
      if num_words_approx > min_num_words then
        terminated = true
      end
    end

  end

  self:resetStates()
  return sampled_string
end


function LM:clearState()
  self.net:clearState()
end
