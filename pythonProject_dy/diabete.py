#from installLibreries import installPackages
from experta import *
from colorama import Fore
from diabetes_data import diabetes_data
from diabetes_ontology import diabetes_ontology

DIABETES_RANDOM_TEST = 11.1
DIABETES_FASTING_TEST = 7

MINIMUM_SKIN_TICKNESS = 10
MAXIMUM_SKIN_TICKNESS = 100


def reset_color():
    print(Fore.RESET)


def valid_response(response: str):
    valid = False
    response = response.lower()

    if response == "si" or response == "no":
        valid = True

    return valid


def valid_random_test_blood_value(test_value: float):
    valid = False

    if test_value > 3.9:
        valid = True

    return valid


def valid_blood_pressure(pressure: int):
    valid = True

    if pressure <= 60 or pressure > 210:
        valid = False

    return valid


class diabete(KnowledgeEngine):

    @DefFacts()
    def _initial_action(self):
        yield Fact(inizio="si")
        self.mean_diabetes_tests = diabetes_data().get_medium_values_diabetes()
        self.number_prints = 0
        self.flag_no_symptoms = 0

    def print_facts(self):
        print("\n\nIl programma si è basato sui i seguenti fatti: \n")
        print(self.facts)

    def _prototype_ask_symptom(self, ask_text: str, fact_declared: Fact):

        print(ask_text)
        response = str(input())

        while valid_response(response) == False:
            print(ask_text)
            response = str(input())
        if response == "si":
            self.declare(fact_declared)

        return response

    @Rule(Fact(inizio="si"))
    def rule_1(self):
        print(Fore.CYAN + "\nInizio della diagnosi...\n")
        reset_color()
        self.declare(Fact(chiedi_sintomi="si"))

    @Rule(Fact(chiedi_esami_glicemia="si"))
    def rule_2(self):
        print("Hai eseguito un test casuale del sangue?")
        casual_blood_test = str(input())

        while valid_response(casual_blood_test) == False:
            print("Hai eseguito un test casuale del sangue?")
            casual_blood_test = str(input())

        print("Hai eseguito un test del sangue a digiuno?")
        fasting_blood_test = str(input())

        while valid_response(fasting_blood_test) == False:
            print("Hai eseguito un test del sangue a digiuno?")
            fasting_blood_test = str(input())

        if casual_blood_test == "si":
            self.declare(Fact(test_casuale_sangue="si"))
        else:
            self.declare(Fact(test_casuale_sangue="no"))

        if fasting_blood_test == "si":
            self.declare(Fact(test_digiuno_sangue="si"))
        else:
            self.declare(Fact(test_digiuno_sangue="no"))


    @Rule(Fact(test_casuale_sangue="si"))
    def rule_3(self):
        print(
            "Inserisci il valore del test espresso in millimoli su litro [mmol/L]")
        test_value = float(input())

        while valid_random_test_blood_value(test_value) == False:
            print("Inserisci il valore del test espresso in millimoli su litro [mmol/L]")
            test_value = float(input())

        if test_value > DIABETES_RANDOM_TEST:
            self.declare(Fact(glicemia_casuale_alta="si"))

        else:
            self.declare(Fact(glicemia_normale="si"))

    @Rule(Fact(test_digiuno_sangue="si"))
    def rule_4(self):
        print(
            "Inserisci il valore del test espresso in millimoli su litro [mmol/L]")
        test_value = float(input())

        while valid_random_test_blood_value(test_value) == False:
            print(
                "Inserisci il valore del test espresso in millimoli su litro [mmol/L]")
            test_value = float(input())

        if test_value > DIABETES_FASTING_TEST:
            self.declare(Fact(glicemia_digiuno_alta="si"))
        else:
            self.declare(Fact(glicemia_normale="si"))

    @Rule(Fact(test_digiuno_sangue="no"))
    def rule_5(self):
        print(Fore.YELLOW + "Prenota gli esami del sangue per avere una diagnosi più certa.")
        reset_color()

    @Rule(Fact(chiedi_sintomi="si"))
    def rule_6(self):

        r1 = self._prototype_ask_symptom("Ti senti molto assetato di solito (sopratutto di notte) ? [si/no]", Fact(molta_sete="si"))
        r2 = self._prototype_ask_symptom("Ti senti molto stanco? [si/no]", Fact(molto_stanco="si"))
        r3 = self._prototype_ask_symptom("Stai perdendo peso e massa muscolare? [si/no]", Fact(perdita_massa="si"))
        r4 = self._prototype_ask_symptom("Senti prurito? [si/no]", Fact(prurito="si"))
        r5 = self._prototype_ask_symptom("Hai la vista offuscata? [si/no]", Fact(vista_offuscata="si"))
        r6 = self._prototype_ask_symptom("Consumi spesso bevande/alimenti zuccherati? [si/no]",Fact(bevande_zuccherate="si"))
        r7 = self._prototype_ask_symptom("Hai fame costantemente? [si/no]", Fact(fame_costante="si"))
        r8 = self._prototype_ask_symptom("Hai spesso la bocca asciutta? [si/no]", Fact(bocca_asciutta="si"))

        if r1 == "no" and r2 == "no" and r3 == "no" and r4 == "no" and r5 == "no" and r6 == "no" and r7 == "no" and r8 == "no":
            self.flag_no_symptoms = 1

        self.declare(Fact(chiedi_imc="si"))

    @Rule(Fact(chiedi_imc="si"))
    def ask_bmi(self):

        medium_bmi_diabetes = self.mean_diabetes_tests['BMI']

        print(Fore.CYAN + "\n\nInserisci l'altezza in centimetri")
        reset_color()
        height = int(input())

        while height < 135 or height > 220:
            print(Fore.CYAN + "Inserisci di nuovo l'altezza in centimetri")
            reset_color()
            height = int(input())

        print(Fore.CYAN + "Inserisci il peso in kilogrammi")
        reset_color()
        weight = int(input())

        while weight < 30 or weight > 250:
            print(Fore.CYAN + "Inserisci di nuovo il peso in kilogrammi")
            reset_color()
            weight = int(input())

        bmi = round(height / (weight * weight), 3)

        if bmi >= medium_bmi_diabetes:
            print(Fore.YELLOW + "Il valore del tuo indice di massa corporea paria a %f e' superiore al valore medio di indice di massa corporea dei diabetici" % bmi)
            reset_color()

    @Rule(Fact(esami_pressione="si"))
    def ask_pressure_exam(self):
        print("Hai fatto l'esame della pressione sanguigna?")
        response = str(input())

        while valid_response(response) == False:
            print("Hai fatto l'esame della pressione sanguigna?")
            response = str(input())

        if response == "si":
            self.declare(Fact(esame_pressione_eseguito="si"))
        else:
            self.declare(Fact(prescrizione_esame_pressione="no"))

    @Rule(Fact(prescrizione_esame_pressione="no"))
    def pressure_exams_book(self):
        print(Fore.YELLOW + "Prenota gli esami della pressione per avere una diagnosi più certa.")
        reset_color()

    @Rule(Fact(esame_pressione_eseguito="si"))
    def pressure_exam(self):

        medium_pressure = self.mean_diabetes_tests['BloodPressure']

        print("Inserisci il valore della pressione sanguigna")
        pressure_value = int(input())

        while valid_blood_pressure(pressure_value) == False:
            print("Inserisci il valore della pressione sanguigna")
            pressure_value = int(input())

        if pressure_value >= medium_pressure:
            self.declare(Fact(diagnosi_pressione_diabete="si"))

        else:
            self.declare(Fact(diagnosi_pressione_normale="si"))

    @Rule(Fact(diagnosi_pressione_normale="si"))
    def normal_blood_pressure(self):
        print(Fore.GREEN + "Il valore della pressione sembra nella norma")
        reset_color()

    @Rule(Fact(diagnosi_pressione_diabete="si"))
    def blood_pressure_diabetes(self):
        print(Fore.YELLOW + "Il valore della pressione e' maggiore o uguale a quella dei diabetici")
        reset_color()

    @Rule(OR(Fact(fame_costante="si"), Fact(bevande_zuccherate="si")))
    def exam_1(self):
        self.declare(Fact(chiedi_esami_glicemia="si"))

    @Rule(OR(Fact(vista_offuscata="si"), Fact(molto_stanco="si"), Fact(bocca_asciutta="si")))
    def exam_2(self):
        self.declare(Fact(esami_pressione="si"))

    @Rule(AND(Fact(molta_sete="si"), Fact(molto_stanco="si"), Fact(perdita_massa="si"), Fact(prurito="si"),
              Fact(vista_offuscata="si"), Fact(bevande_zuccherate="si"), Fact(fame_costante="si"),
              Fact(bocca_asciutta="si")))
    def all_diabetes_symptoms(self):
        print(Fore.YELLOW + "Sembra che tu abbia TUTTI i sintomi del diabete")
        reset_color()
        self.declare(Fact(tutti_sintomi="si"))
        self.declare(Fact(chiedi_esami_glicemia="si"))
        self.declare(Fact(esami_pressione="si"))

    @Rule(AND(Fact(molta_sete="si"), Fact(molto_stanco="si"), Fact(perdita_massa="si"), Fact(prurito="si"),
              Fact(vista_offuscata="si"), Fact(bevande_zuccherate="si"), Fact(fame_costante="si"),
              Fact(bocca_asciutta="si")), Fact(diagnosi_pressione_diabete="si"), Fact(glicemia_digiuno_alta="si"),
          Fact(glicemia_casuale_alta="si"), Fact(diagnosi_pressione_diabete="si"), Fact(insulina_alta_diabete="si"))
    def all_diabetes_diagnosis_3(self):
        print(Fore.RED + "Hai sicuramente il diabete")
        reset_color()
        self.declare(Fact(diabete_tutti_sintomi="si"))

    @Rule(OR(Fact(prurito="si"), Fact(perdita_massa="si")))
    def ask_itching_test(self):
        print("Hai fatto un test per misurare lo spessore della piega cutanea del tricipite?")
        response = str(input())

        while valid_response(response) == False:
            print("Hai fatto un test per misurare lo spessore della piega cutanea del tricipite?")
            response = str(input())

        if response == "si":
            self.declare(Fact(esame_pelle="si"))

        else:
            self.declare(Fact(esame_pelle="no"))


    @Rule(Fact(esame_pelle="no"))
    def skin_exams_book(self):

        print(Fore.YELLOW + "Prenota gli esami per avere una diagnosi più certa.")
        reset_color()

    @Rule(Fact(esame_pelle="si"))
    def itching_test(self):

        medium_diabetes_thickness = self.mean_diabetes_tests['SkinThickness']
        print("Hai detto di aver fatto l'esame per misurare lo spessore della piega cutanea del tricipite")

        print("Inserisci il valore in millimetri")
        skin_thick = int(input())

        while skin_thick < MINIMUM_SKIN_TICKNESS or skin_thick > MAXIMUM_SKIN_TICKNESS:
            print("Inserisci di nuovo il valore in millimetri")
            skin_thick = int(input())

        if skin_thick >= medium_diabetes_thickness:
            print(Fore.YELLOW + "Lo spessore della pelle e' maggiore o uguale a quello dei diabetici, prova a fare altri esami!")
            reset_color()

    @Rule(OR(Fact(fame_costante="si"), Fact(bevande_zuccherate="si")))
    def ask_insulin_exam(self):

        print("Hai eseguito un test per misurare il valore di insulina?")
        response = str(input())

        while valid_response(response) == False:
            print("Hai eseguito un test per misurare il valore di insulina?")
            response = str(input())

        if response == "si":
            self.declare(Fact(test_insulina="si"))

        else:
            self.declare(Fact(test_insulina="no"))

    @Rule(Fact(test_insulina="si"))
    def insulin_exam(self):

        medium_insulin_diabetes = self.mean_diabetes_tests['Insulin']

        print("Insersci il valore dell'insulina espresso in mu U/ml")
        insulin_value = float(input())

        while insulin_value < 0 or insulin_value > 700:
            print("Insersci il valore dell'insulina espresso in mu U/ml")
            insulin_value = float(input())

        if insulin_value >= medium_insulin_diabetes:
            self.declare(Fact(insulina_alta_diabete="si"))

    @Rule(Fact(test_insulina="no"))
    def insulin_prescription(self):

        print(Fore.YELLOW + "Dovresti prenotare gli esami per vedere il livello di insulina.")
        reset_color()

        #self._prototype_lab_booking("gli esami dell' insulina", self.lab_insulin_analysis)

    @Rule(AND(Fact(insulina_alta_diabete="si"), NOT(Fact(diagnosi_diabete_incerta="si"))))
    def diagnosis_4(self):
        print(Fore.RED + "Hai il diabete!")
        reset_color()
        self.declare(Fact(diagnosi_diabete="si"))

    @Rule(Fact(prescrizione_esami_sangue="si"))
    def prescription_1(self):
        print(Fore.YELLOW + "Dovresti fare gli esami per misurare la glicemia nel sangue!")
        reset_color()


    @Rule(Fact(glicemia_normale="si"))
    def normal_blood_glucose(self):
        print(Fore.GREEN + "La glicemia e' nella norma.")
        reset_color()

    @Rule(NOT(AND(Fact(molta_sete="si"), Fact(molto_stanco="si"), Fact(perdita_massa="si"), Fact(prurito="si"),
                  Fact(vista_offuscata="si"), Fact(bevande_zuccherate="si"), Fact(fame_costante="si"),
                  Fact(bocca_asciutta="si"))))
    def not_symptoms(self):

        if self.number_prints == 0 and self.flag_no_symptoms == 1:
            print(Fore.GREEN + "\n\nNon hai alcun sintomo del diabete!")
            self.declare(Fact(niente_sintomi="si"))
            reset_color()
            self.number_prints = self.number_prints + 1

    @Rule(NOT(OR(Fact(diagnosi_diabete="si"), Fact(diabete_tutti_sintomi="si"), Fact(tutti_sintomi="si"))))
    def intermediate_case(self):

        if self.flag_no_symptoms != 1:
            print(Fore.YELLOW + "Potresti avere il diabete, rivolgiti da un medico per fare accertamenti!")
            self.declare(Fact(diagnosi_diabete_incerta="si"))
            reset_color()

def main_agent():
    expert_agent = diabete()
    expert_agent.reset()
    expert_agent.run()
    expert_agent.print_facts()


def main_ontology():
    do = diabetes_ontology()

    do.get_symptoms_descriptions()
    symptoms, keys_symptoms = do.print_symptoms()

    print("\nSeleziona il sintomo di cui vuoi conosere la descrizione, inserisci il numero del sintomo")
    symptom_number = int(input())

    while symptom_number not in symptoms.keys():
        print("\nSeleziona il sintomo di cui vuoi conosere la descrizione, inserisci il numero del sintomo")
        symptom_number = int(input())

    print("Sintomo: %s, descrizione: %s" % (keys_symptoms[symptom_number], " ".join(symptoms[symptom_number])))


if __name__ == '__main__':

    exit_program = False

    print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(Fore.MAGENTA +
        "\n\nIl diabete è una malattia che si caratterizza per la presenza di quantità eccessive di glucosio (zucchero) \n"
        "nel sangue. L’eccesso di glucosio, noto con il termine di iperglicemia, può essere causato da un’insufficiente produzione \n"
        "di insulina o da una sua inadeguata azione; l’insulina è l’ormone che regola il livello di glucosio nel sangue. \n"
        "Le diagnosi di diabete stanno aumentando a livello mondiale e nel nostro Paese colpisce più di 3,5 milioni di persone. \n")
    while exit_program == False:

        print(Fore.LIGHTMAGENTA_EX +
            "--------------->MENU<---------------\n\n[1] Scopri i sintomi più comuni del diabete\n[2] Esegui una diagnosi per scoprire se potresti soffrire di diabete di tipo 1\n[3] Esci dal programma")
        user_choose = None

        try:
            user_choose = int(input())

        except ValueError:
            exit_program = True

        if user_choose == 1:
            main_ontology()

        elif user_choose == 2:
            main_agent()

        else:
            print(Fore.MAGENTA + "\n\n\n\n\nUscita dal programma...")
            exit_program = True

        print("\n\n")