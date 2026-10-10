"""BERT post-norm cross-encoder, exact GELU, pooler and scalar head."""
import mlx.core as mx
import mlx.nn as nn


def gelu(x):
    return .5*x*(1+mx.erf(x/(2**.5)))


class Embeddings(nn.Module):
    def __init__(self,c):
        self.word_embeddings = nn.Embedding(c['vocab_size'],384)
        self.position_embeddings = nn.Embedding(512,384)
        self.token_type_embeddings = nn.Embedding(2,384)
        self.LayerNorm = nn.LayerNorm(384,eps=c['layer_norm_eps'])

    def __call__(self, ids, segments):
        return self.LayerNorm(self.word_embeddings(ids) + self.token_type_embeddings(segments) +
                              self.position_embeddings(mx.arange(ids.shape[1]))[None])


class SelfAttention(nn.Module):
    def __init__(self):
        self.query,self.key,self.value = [nn.Linear(384,384) for _ in range(3)]

    def __call__(self,x,mask):
        shape = x.shape
        q,k,v = [projection(x).reshape(shape[0],shape[1],12,32).transpose(0,2,1,3)
                 for projection in (self.query,self.key,self.value)]
        return mx.fast.scaled_dot_product_attention(q,k,v,scale=32**-.5,mask=mask).transpose(0,2,1,3).reshape(shape)


class Output(nn.Module):
    def __init__(self, width, eps):
        self.dense = nn.Linear(width,384)
        self.LayerNorm = nn.LayerNorm(384,eps=eps)
        self.custom = False

    def __call__(self, x, residual):
        branch = self.dense(x)
        if self.custom:
            from .metal_kernel import residual_norm
            return residual_norm(residual,branch,self.LayerNorm.weight,self.LayerNorm.bias,self.LayerNorm.eps)
        return self.LayerNorm(branch+residual)


class Attention(nn.Module):
    def __init__(self,eps):
        self.self = SelfAttention()
        self.output = Output(384,eps)

    def __call__(self,x,mask):
        return self.output(self.self(x,mask),x)


class Intermediate(nn.Module):
    def __init__(self):
        self.dense = nn.Linear(384,1536)

    def __call__(self,x):
        return gelu(self.dense(x))


class Layer(nn.Module):
    def __init__(self,eps):
        self.attention = Attention(eps)
        self.intermediate = Intermediate()
        self.output = Output(1536,eps)

    def __call__(self,x,mask):
        x = self.attention(x,mask)
        return self.output(self.intermediate(x),x)


class Encoder(nn.Module):
    def __init__(self,eps):
        self.layer = [Layer(eps) for _ in range(6)]


class Pooler(nn.Module):
    def __init__(self):
        self.dense = nn.Linear(384,384)

    def __call__(self,x):
        return mx.tanh(self.dense(x[:,0]))


class Bert(nn.Module):
    def __init__(self,c):
        self.embeddings = Embeddings(c)
        self.encoder = Encoder(c['layer_norm_eps'])
        self.pooler = Pooler()


class Model(nn.Module):
    def __init__(self,c):
        self.bert = Bert(c)
        self.classifier = nn.Linear(384,1)

    def __call__(self,input_ids,attention_mask,token_type_ids,trace=False):
        x = self.bert.embeddings(input_ids,token_type_ids)
        states = [x]
        mask = (attention_mask!=0)[:,None,None,:]
        for layer in self.bert.encoder.layer:
            x = layer(x,mask)
            states.append(x)
        pool = self.bert.pooler(x)
        scores = self.classifier(pool)[:,0]
        return (states,pool,scores) if trace else scores
