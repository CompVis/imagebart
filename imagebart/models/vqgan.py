from taming.models.vqgan import VQModel


class VQGANWrapper(VQModel):

    def __init__(self,embed_dim,*args,**kwargs):
        super().__init__(embed_dim=embed_dim,*args,**kwargs)
        self.embed_dim = embed_dim

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h