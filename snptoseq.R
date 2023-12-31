
library(data.table)


args<-commandArgs(trailingOnly = TRUE)

jobname="HGMD"
snpfile="HGMD.bed"
ext=as.integer(500)
genome="BSgenome.Hsapiens.UCSC.hg19"
message('jobname: ',jobname)

message('snpfile:', snpfile)

message('window ext: ',ext)

message('ref  genome: ',genome)
if(genome=='BSgenome.Hsapiens.UCSC.hg19'){
  library("BSgenome.Hsapiens.UCSC.hg19")
  bsgenome=BSgenome.Hsapiens.UCSC.hg19
}else if(genome=='BSgenome.Hsapiens.UCSC.hg38'){
  library("BSgenome.Hsapiens.UCSC.hg38")
  bsgenome=BSgenome.Hsapiens.UCSC.hg38
}
snp=fread(snpfile)
snp=as.data.frame(snp)
colnames(snp)=c('chr','start','end','label')

if(length(grep('chr',snp$chr))==0) snp$chr=paste0('chr',snp$chr)
snp.gr=makeGRangesFromDataFrame(snp)

seqs=getSeq(bsgenome,seqnames(snp.gr),start(snp.gr)-ext,end(snp.gr)+ext)

length(grep('N',seqs))

seqs=DNAStringSet(gsub('N','A',seqs))

export(seqs,paste0(jobname,'.fasta'))

write.table(snp,file=paste0(jobname,'.bed'),row.names = F,quote=F,sep='\t')


# Rscript --vanilla snptoseq.R  COSMIC  /Users/COSMIC 500 hg19








